use chrono::prelude::*;
use clap::Parser;
use image::ImageReader;
use rand::Rng;
use rayon::prelude::*;
use rumpus::filter::DopFilter;
use rumpus::prelude::*;
use rumpus::ray::RayFrame;
use serde::{Deserialize, Serialize};
use sguaba::{Bearing, engineering::Orientation, systems::Wgs84, vector};
use std::{io::Read, path::PathBuf};
use uom::{
    ConstZero,
    si::{
        angle::{degree, radian},
        f64::{Angle, Length},
        length::meter,
        ratio::ratio,
    },
};

#[derive(Parser)]
struct Cli {
    image: PathBuf,
    params: PathBuf,
    output: PathBuf,
}

#[derive(Serialize, Deserialize)]
struct SimulationParams {
    pixel_size: Length,
    focal_length: Length,
    latitude: Angle,
    longitude: Angle,
    time: DateTime<Utc>,
    min_dop: f64,
    max_iters: usize,
    /// The rate at which the current estimate descends to the optimal estimate.
    ///
    /// A larger learning rate can decrease the number of steps to the optimal estimate, but may lead to oscillation
    /// about the optimal estimate, preventing convergence.
    /// The best learning rate is the largest rate that still converges.
    learning_rate: f64,

    /// The threshold on the magnitude of the gradient that is considered the optimal estimate.
    convergence_threshold: f64,
}

#[derive(Serialize)]
struct Candidate {
    yaw: f64,
    pitch: f64,
    roll: f64,
    /// The x coordinate of the up vector when oriented at yaw, pitch, roll.
    optical_axis_east: f64,
    /// The y coordinate of the up vector when oriented at yaw, pitch, roll.
    optical_axis_north: f64,
    loss: f64,
    gradient: f64,
}

fn main() {
    let args = Cli::parse();
    let params = parse_params(&args.params).expect("readable and parsable params");

    let raw_image = ImageReader::open(&args.image)
        .expect("a valid image path")
        .decode()
        .expect("a supported image format")
        .into_luma8();

    let (width, height) = raw_image.dimensions();
    let rays: Vec<Ray<SensorFrame>> =
        IntensityImage::from_bytes(width as u16, height as u16, &raw_image.into_raw())
            .expect("image dimensions are even")
            .rays(params.pixel_size, params.pixel_size)
            .ray_filter(DopFilter::new(params.min_dop))
            .collect();

    let ray_count = rays.len();
    if ray_count == 0 {
        println!("no rays met dop threshold");
        std::process::exit(0);
    }

    println!("selected {} rays", ray_count);

    let lens =
        Lens::from_focal_length(params.focal_length).expect("focal length is greater than zero");

    let model = SkyModel::from_wgs84_and_time(
        Wgs84::builder()
            .latitude(params.latitude)
            .expect("latitude is between -90 and 90")
            .longitude(params.longitude)
            .altitude(Length::ZERO)
            .build(),
        params.time,
    );

    let mut rng = rand::rng();
    let mut orientation = Orientation::<CameraEnu>::tait_bryan_builder()
        .yaw(Angle::new::<degree>(rng.random_range(0.0..360.0)))
        .pitch(Angle::new::<degree>(0.0))
        .roll(Angle::new::<degree>(0.0))
        .build();

    let mut csv = csv::Writer::from_path(&args.output).unwrap();

    let mut iters = 0;
    loop {
        iters += 1;
        if iters > params.max_iters {
            println!("reached max_iters");
            break;
        }

        // Construct a camera at the new orientation.
        let cam = Camera::new(lens.clone(), orientation.clone());

        // Find the zenith coordinate in CameraFrd.
        let zenith_coord = cam
            .trace_from_sky(
                Bearing::<CameraEnu>::builder()
                    .azimuth(Angle::ZERO)
                    .elevation(Angle::HALF_TURN / 2.)
                    .expect("elevation is on range -90 to 90")
                    .build(),
            )
            .expect("zenith is always above the horizon");

        struct RayInfo<Frame: RayFrame> {
            difference: Aop<Frame>,
            partial_derivative: Aop<Frame>,
            weight: f64,
        }

        let ray_info: Vec<RayInfo<_>> = rays
            .par_iter()
            .filter_map(|ray_sensor| {
                // Model a ray with the same CameraFrd coordinate as the measured ray.
                let ray_bearing = cam
                    .trace_from_sensor(*ray_sensor.coord())
                    .expect("ray coordinate should always have Z of zero");

                // Ignore rays from below the horizon.
                let modelled_aop = model.aop(ray_bearing)?;
                let modelled_ray_global = Ray::new(*ray_sensor.coord(), modelled_aop, Dop::zero());
                let partial_derivative = delta_aop(&model, ray_bearing)?;

                // Transform the measured ray from the sensor frame into the global frame.
                let ray_global = ray_sensor
                    .into_global_frame(zenith_coord.clone())
                    // Camera trace_from_sky always returns a coordinate with a zenith of zero which enforces this expect.
                    .expect("zenith coord has a Z of zero");

                let difference = *modelled_ray_global.aop() - *ray_global.aop();
                let weight = 1. / (*ray_global.dop()).into_inner();

                Some(RayInfo {
                    difference,
                    partial_derivative,
                    weight,
                })
            })
            .collect();

        println!("computed ray info");

        let loss = ray_info
            .par_iter()
            .map(|ri| {
                let sq_diff = ri.difference.into_inner().get::<radian>().powf(2.);
                let weighted_sq_diff = ri.weight * sq_diff;

                weighted_sq_diff
            })
            // Take the mean of the weighted, squared differences.
            .sum::<f64>()
            / rays.len() as f64;

        let gradient = ray_info
            .par_iter()
            .map(|ri| {
                let weighted_delta = ri.weight
                    * ri.difference.into_inner().get::<radian>()
                    * ri.partial_derivative.into_inner().get::<radian>();

                weighted_delta
            })
            .sum::<f64>()
            / rays.len() as f64
            * 2.;

        println!("loss {}, gradient {}", loss, gradient);

        // Create a unit vector along the camera's optical axis.
        let optical_axis_frd = vector!(
            f = Length::new::<meter>(0.0),
            r = Length::new::<meter>(0.0),
            d = Length::new::<meter>(1.0);
            in CameraFrd
        );

        // Transform the unit vector by orientation.
        let camera_frd_to_enu = unsafe { orientation.map_as_zero_in::<CameraFrd>() }.inverse();
        let optical_axis_enu = camera_frd_to_enu.transform(optical_axis_frd);

        let (yaw, pitch, roll) = orientation.to_tait_bryan_angles();
        csv.serialize(Candidate {
            yaw: yaw.get::<degree>(),
            pitch: pitch.get::<degree>(),
            roll: roll.get::<degree>(),
            optical_axis_east: optical_axis_enu.enu_east().get::<meter>(),
            optical_axis_north: optical_axis_enu.enu_north().get::<meter>(),
            loss,
            gradient,
        })
        .unwrap();

        if gradient.abs() < params.convergence_threshold {
            println!("estimate converged");
            break;
        }

        // Gradient is the expected delta loss per delta yaw.
        // We want to move towards a yaw that minimizes the loss.
        // If delta loss is positive that implies increasing yaw will increase loss.
        // If delta loss is negative that implies decreasing yaw will increase loss.
        // We want to do the opposite.
        // If we haven't converged, then compute the next orientation to check.
        // The next orientation should be the current orientation, but shifted by:
        //   O' = O - learning_rate * gradient
        // To start, we only vary the yaw and assume pitch and roll are zero.
        //   O'.yaw = O.yaw - learning_rate * gradient
        let (yaw, _pitch, _roll) = orientation.to_tait_bryan_angles();
        let delta_yaw = Angle::new::<radian>(params.learning_rate * gradient);
        orientation = Orientation::<CameraEnu>::tait_bryan_builder()
            .yaw(yaw - delta_yaw)
            .pitch(Angle::new::<degree>(0.0))
            .roll(Angle::new::<degree>(0.0))
            .build();
    }
}

fn parse_params(path: &PathBuf) -> Option<SimulationParams> {
    let mut buffer = String::new();
    std::fs::File::open(path)
        .ok()?
        .read_to_string(&mut buffer)
        .ok()?;
    toml::from_str(&buffer).ok()
}

///
/// \frac{c\left(bc\sin ^2\left(x-d\right)-\cos \left(x-d\right)\left(-bc\cos \left(x-d\right)+a\right)\right)}{\left(-bc\cos \left(x-d\right)+a\right)^2+c^2\sin ^2\left(x-d\right)}
pub fn delta_aop(model: &SkyModel, bearing: Bearing<CameraEnu>) -> Option<Aop<GlobalFrame>> {
    if bearing.elevation() < Angle::ZERO {
        return None;
    }

    let solar_azimuth = model.solar_bearing().azimuth();
    let solar_zenith = Angle::HALF_TURN / 2. - model.solar_bearing().elevation();
    let azimuth = bearing.azimuth();
    let zenith = Angle::HALF_TURN / 2. - bearing.elevation();

    let a = (zenith.sin() * solar_zenith.cos()).get::<ratio>();
    let b = zenith.cos().get::<ratio>();
    let c = solar_zenith.sin().get::<ratio>();
    let d = (azimuth - solar_azimuth).get::<radian>();

    let angle = c * (b * c * d.sin().powf(2.) - d.cos() * (-1. * b * c * d.cos() + a))
        / (-1. * b * c * d.cos() + a).powf(2.)
        + c.powf(2.) * d.sin().powf(2.);

    Some(Aop::from_angle_wrapped(Angle::new::<radian>(angle)))
}

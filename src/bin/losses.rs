use chrono::prelude::*;
use clap::Parser;
use image::ImageReader;
use rand::Rng;
use rayon::prelude::*;
use rumpus::light::filter::DopFilter;
use rumpus::prelude::*;
use serde::{Deserialize, Serialize};
use sguaba::{Bearing, engineering::Orientation, systems::Wgs84, vector};
use std::{io::Read, path::PathBuf};
use uom::{
    ConstZero,
    si::{
        angle::{degree, radian},
        f64::{Angle, Length},
        length::meter,
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
    let mut csv = csv::Writer::from_path(&args.output).unwrap();
    for _ in 0..params.max_iters {
        let orientation = Orientation::<CameraEnu>::tait_bryan_builder()
            .yaw(Angle::new::<degree>(rng.random_range(0.0..360.0)))
            .pitch(Angle::new::<degree>(0.0))
            .roll(Angle::new::<degree>(0.0))
            .build();

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

        let loss = rays
            .par_iter()
            .filter_map(|ray_sensor| {
                // Model a ray with the same CameraFrd coordinate as the measured ray.
                let ray_bearing = cam
                    .trace_from_sensor(*ray_sensor.coord())
                    .expect("ray coordinate should always have Z of zero");

                // Ignore rays from below the horizon.
                let modelled_aop = model.aop(ray_bearing)?;
                let modelled_ray_global = Ray::new(*ray_sensor.coord(), modelled_aop, Dop::zero());

                // Transform the measured ray from the sensor frame into the global frame.
                let ray_global = ray_sensor
                    .into_global_frame(zenith_coord.clone())
                    // Camera trace_from_sky always returns a coordinate with a zenith of zero which enforces this expect.
                    .expect("zenith coord has a Z of zero");

                // Compute the weighted, squared difference between the modelled ray and the measured ray.
                let delta = *modelled_ray_global.aop() - *ray_global.aop();
                let sq_diff = delta.into_inner().get::<radian>().powf(2.);
                let weight = 1. / (*ray_global.dop()).into_inner();
                let weighted_sq_diff = weight * sq_diff;

                Some(weighted_sq_diff)
            })
            // Take the mean of the weighted, squared differences.
            .sum::<f64>()
            / rays.len() as f64;

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
        })
        .unwrap();
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

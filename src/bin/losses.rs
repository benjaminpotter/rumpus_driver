use chrono::prelude::*;
use clap::Parser;
use image::ImageReader;
use rand::Rng;
use rumpus::prelude::*;
use rumpus::{estimator::pattern_match::Searcher, light::filter::DopFilter};
use serde::{Deserialize, Serialize};
use sguaba::{Bearing, engineering::Orientation, systems::Wgs84};
use std::{
    io::{Read, Write},
    path::PathBuf,
};
use uom::{
    ConstZero,
    si::{
        angle::{degree, radian},
        f64::{Angle, Length},
    },
};

#[derive(Parser)]
struct Cli {
    image: PathBuf,
    params: PathBuf,
    #[arg(short, long)]
    output: Option<PathBuf>,
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

    let mut csv = csv::Writer::from_path(&args.output.unwrap()).unwrap();
    RandomSearch::new(rand::rng())
        .orientations()
        .take(params.max_iters)
        .for_each(|orientation| {
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
                .iter()
                .filter_map(|ray| {
                    // Model a ray with the same CameraFrd coordinate as the
                    // measured ray.
                    let ray_bearing = cam
                        .trace_from_sensor(*ray.coord())
                        .expect("ray coordinate should always have Z of zero");
                    // Ignore rays from below the horizon.
                    let modelled_aop = model.aop(ray_bearing)?;
                    let modelled_ray_global = Ray::new(*ray.coord(), modelled_aop, Dop::zero());

                    // Transform the modelled ray from the global frame into
                    // the sensor frame.
                    let modelled_ray_sensor = modelled_ray_global
                        .into_sensor_frame(zenith_coord.clone())
                        // Camera trace_from_sky always returns a coordinate
                        // with a zenith of zero which enforces this expect.
                        .expect("zenith coord is has Z of zero");

                    // Compute the weighted, squared difference between the
                    // modelled ray and the measured ray.
                    let delta = *ray.aop() - *modelled_ray_sensor.aop();
                    let sq_diff = delta.into_inner().get::<radian>().powf(2.);
                    let weight = 1. / (*ray.dop()).into_inner();
                    let weighted_sq_diff = weight * sq_diff;

                    Some(weighted_sq_diff)
                })
                // Take the mean of the weighted, squared differences.
                .sum::<f64>()
                / rays.len() as f64;

            let (yaw, pitch, roll) = orientation.to_tait_bryan_angles();
            csv.serialize(Candidate {
                yaw: yaw.get::<degree>(),
                pitch: pitch.get::<degree>(),
                roll: roll.get::<degree>(),
                loss,
            })
            .unwrap();
        });
}

fn parse_params(path: &PathBuf) -> Option<SimulationParams> {
    let mut buffer = String::new();
    std::fs::File::open(path)
        .ok()?
        .read_to_string(&mut buffer)
        .ok()?;
    toml::from_str(&buffer).ok()
}

struct RandomSearch<R> {
    rng: R,
}

impl<R> RandomSearch<R> {
    fn new(rng: R) -> Self {
        Self { rng }
    }
}

impl<R: Rng> Searcher for RandomSearch<R> {
    type Iter = RandomSearchIter<R>;

    fn orientations(self) -> Self::Iter {
        RandomSearchIter { rng: self.rng }
    }
}

struct RandomSearchIter<R> {
    rng: R,
}

impl<R: Rng> Iterator for RandomSearchIter<R> {
    type Item = Orientation<CameraEnu>;

    fn next(&mut self) -> Option<Self::Item> {
        Some(
            Orientation::<CameraEnu>::tait_bryan_builder()
                .yaw(Angle::new::<degree>(self.rng.random_range(0.0..360.0)))
                .pitch(Angle::new::<degree>(self.rng.random_range(-5.0..5.0)))
                .roll(Angle::new::<degree>(self.rng.random_range(-5.0..5.0)))
                .build(),
        )
    }
}

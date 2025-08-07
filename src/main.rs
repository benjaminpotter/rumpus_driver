use chrono::prelude::*;
use clap::Parser;
use image::ImageReader;
use nalgebra::{Rotation3, Vector2};
use rumpus::prelude::*;
use std::path::PathBuf;

#[derive(Parser)]
struct Cli {
    image_path: PathBuf,
}

fn main() {
    let args = Cli::parse();

    let raw_image = ImageReader::open(args.image_path)
        .unwrap()
        .decode()
        .unwrap()
        .into_luma8();

    let (width, height) = raw_image.dimensions();
    let estimate = IntensityImage::from_bytes(width, height, &raw_image.into_raw())
        .unwrap()
        .rays(&RaySensor::new(
            Vector2::new(3.45 * 0.001 * 2.0, 3.45 * 0.001 * 2.0),
            Vector2::new(2448.0 / 2.0, 2048.0 / 2.0),
        ))
        .estimate(PatternMatch::new(
            Lens::new(8.0),
            RayleighModel::new(
                Position {
                    lat: 44.2187,
                    lon: -76.4747,
                },
                "2025-06-13T16:26:47+00:00"
                    .parse::<DateTime<Utc>>()
                    .unwrap(),
            ),
            StochasticSearch::try_new(
                Orientation::new(Rotation3::from_euler_angles(-1.0, -1.0, 0.0)),
                Orientation::new(Rotation3::from_euler_angles(1.0, 1.0, 360.0)),
                rand::rng(),
            )
            .unwrap(),
            10,
        ));

    println!("yaw");
    println!("{}", estimate.euler_angles().2.to_degrees());
}

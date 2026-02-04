use std::{
    fs::File,
    io::{BufReader, Cursor},
    path::PathBuf,
};

use anyhow::Result;

use clap::Parser;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Path to .glb file
    input: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let file = File::open(args.input)?;
    let mut src = BufReader::new(file);

    let model = models::Gltf::parse(&mut src)?;
    println!("{:#?}", model.header);
    println!("{:#?}", model.meta);
    Ok(())
}

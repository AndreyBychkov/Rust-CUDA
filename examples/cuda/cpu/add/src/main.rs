use cust::prelude::*;
use nanorand::{Rng, WyRand};
use std::error::Error;
use std::time::Instant;

const N: usize = 1024;
const N2: usize = (N * N) as usize;

static PTX: &str = include_str!("../../../resources/add.ptx");

pub(crate) fn matmul_gpu(lhs: &Vec<f32>, rhs: &Vec<f32>) -> Result<Vec<f32>, Box<dyn Error>> {

    let _ctx = cust::quick_init()?;
    let module = Module::from_ptx(PTX, &[])?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    let lhs_gpu = lhs.as_slice().as_dbuf()?;
    let rhs_gpu = rhs.as_slice().as_dbuf()?;

    let mut out = vec![0.0f32; N2];
    let out_buf = out.as_slice().as_dbuf()?;

    let func = module.get_function("my_matmul_tiled_2d")?;
    // let (_, block_size) = func.suggested_launch_configuration(0, 0.into())?;
    let block_size = 16;
    let grid_size = (N as u32 + block_size - 1) / block_size;
    let dim_block = (block_size, block_size, 1);
    let dim_grid = (grid_size, grid_size, 1);

    println!(
        "using {} blocks and {} threads per block ({} threads total)",
        grid_size*grid_size, block_size*block_size, grid_size*grid_size*block_size*block_size
    );

    let start = Instant::now();
    unsafe {
        launch!(
            func<<<dim_grid, dim_block, 0, stream>>>(
                lhs_gpu.as_device_ptr(),
                lhs_gpu.len(),
                rhs_gpu.as_device_ptr(),
                rhs_gpu.len(),
                out_buf.as_device_ptr(),
                N
            )
        ).expect("Launch was wrong");
    }

    stream.synchronize()?;
    let elapsed = start.elapsed();
    println!("Spent on kernel = {}s,\tPerf = {} Gflops", elapsed.as_secs_f32(), 2.0*1e-9* (N*N*N) as f32);

    out_buf.copy_to(&mut out)?;
    Ok(out)
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut wyrand = WyRand::new();
    let mut lhs = vec![2.0f32; N2];
    wyrand.fill(&mut lhs);
    let mut rhs = vec![0.0f32; N2];
    wyrand.fill(&mut rhs);

    // let out_cpu = matmul_cpu(&lhs, &rhs);
    let start = Instant::now();
    let out_gpu = matmul_gpu(&lhs, &rhs).expect("Problem with gpu code");
    let elapsed = start.elapsed();
    println!("{}s elapsed", elapsed.as_secs_f32());
    // for (c, g) in out_cpu.iter().zip(out_gpu.iter()) {
    //     abs_diff_eq!(c, g);
    // }
    println!("Ok!");
    Ok(())
}

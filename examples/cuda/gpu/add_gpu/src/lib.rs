#![cfg_attr(
target_os = "cuda",
no_std,
feature(register_attr),
register_attr(nvvm_internal)
)]

use cuda_std::prelude::*;
use cuda_std::{shared_array, my_shared_array};
use cuda_std::thread::{block_dim_x, block_dim_y, block_idx_x, block_idx_y, sync_threads, thread_idx_x, thread_idx_y};
use core::mem::{MaybeUninit, transmute};
use core::cell::UnsafeCell;

#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn add(a: &[f32], b: &[f32], c: *mut f32) {
    let idx = thread::index_1d() as usize;
    if idx < a.len() {
        let elem = &mut *c.add(idx);
        *elem = a[idx] + b[idx];
    }
}

const TILE_SZ: usize = 16;

#[kernel]
pub unsafe fn my_matmul_tiled(a: &[f32], b: &[f32], out: *mut f32, n: usize) {
    let row = (block_idx_y() * block_dim_y() + thread_idx_y()) as usize;
    let col = (block_idx_x() * block_dim_x() + thread_idx_x()) as usize;
    let thr_idx_y = thread_idx_y() as usize;
    let thr_idx_x = thread_idx_x() as usize;

    let mut tile_a = my_shared_array![f32; TILE_SZ * TILE_SZ];
    let mut tile_b = my_shared_array![f32; TILE_SZ * TILE_SZ];

    // let mut foo = &mut MaybeUninit::<[f32; 100]>::uninit() as *mut MaybeUninit<[f32; 100]>;
    // let b = *foo.as_mut().unwrap().as_mut_ptr();
    // let a = foo.as_slice();
    // let c = foo.as_mut().unwrap();
    //
    // let mut bar = UnsafeCell::new(MaybeUninit::<[f32; 100]>::uninit());

    let mut sum = 0f32;
    let block_n = (TILE_SZ + n - 1) / TILE_SZ;
    for i in 0..block_n {
        tile_a[thr_idx_y * TILE_SZ + thr_idx_x] =
            if TILE_SZ * i + thr_idx_x < n && row < n {
                a[row * n + i * TILE_SZ + thr_idx_x]
            } else {
                0f32
            };

        tile_b[thr_idx_y * TILE_SZ + thr_idx_x] =
            if TILE_SZ * i + thr_idx_y < n && col < n {
                b[n * (i * TILE_SZ + thr_idx_y) + col]
            } else {
                0f32
            };

        sync_threads();
        for j in 0..TILE_SZ {
            sum += tile_a[thr_idx_y * TILE_SZ + j] * tile_b[j * TILE_SZ + thr_idx_x];
        }
        sync_threads();
    }

    // let out_place = &mut *out;
    // *out_place = sum;

    if row < n && col < n {
        *out.add(row * n + col) = sum;
    }
}

#[kernel]
pub unsafe fn my_matmul_tiled_2d(a: &[f32], b: &[f32], out: *mut f32, n: usize) {
    let row = (block_idx_y() * block_dim_y() + thread_idx_y()) as usize;
    let col = (block_idx_x() * block_dim_x() + thread_idx_x()) as usize;
    let thr_idx_y = thread_idx_y() as usize;
    let thr_idx_x = thread_idx_x() as usize;

    let mut tile_a = my_shared_array![f32; [TILE_SZ; TILE_SZ]];
    let mut tile_b = my_shared_array![f32; [TILE_SZ; TILE_SZ]];

    let mut sum = 0f32;
    let block_n = (TILE_SZ + n - 1) / TILE_SZ;
    for i in 0..block_n {
        tile_a[thr_idx_y][thr_idx_x] =
            if TILE_SZ * i + thr_idx_x < n && row < n {
                a[row * n + i * TILE_SZ + thr_idx_x]
            } else {
                0f32
            };

        tile_b[thr_idx_y][thr_idx_x] =
            if TILE_SZ * i + thr_idx_y < n && col < n {
                b[n * (i * TILE_SZ + thr_idx_y) + col]
            } else {
                0f32
            };

        sync_threads();
        for j in 0..TILE_SZ {
            sum += tile_a[thr_idx_y][j] * tile_b[j][thr_idx_x];
        }
        sync_threads();
    }

    if row < n && col < n {
        *out.add(row * n + col) = sum;
    }
}

#[kernel]
pub unsafe fn matmul_tiled(a: &[f32], b: &[f32], out: *mut f32, n: usize) {
    let row = (block_idx_y() * block_dim_y() + thread_idx_y()) as usize;
    let col = (block_idx_x() * block_dim_x() + thread_idx_x()) as usize;
    let thr_idx_y = thread_idx_y() as usize;
    let thr_idx_x = thread_idx_x() as usize;

    let mut tile_a = shared_array![f32; TILE_SZ * TILE_SZ];
    let mut tile_b = shared_array![f32; TILE_SZ * TILE_SZ];

    let mut sum = 0f32;
    let block_n = (TILE_SZ + n - 1) / TILE_SZ;
    for i in 0..block_n {
        *tile_a.add(thr_idx_y * TILE_SZ + thr_idx_x) =
            if TILE_SZ * i + thr_idx_x < n && row < n {
                a[row * n + i * TILE_SZ + thr_idx_x]
            } else {
                0f32
            };

        *tile_b.add(thr_idx_y * TILE_SZ + thr_idx_x) =
            if TILE_SZ * i + thr_idx_y < n && col < n {
                b[n * (i * TILE_SZ + thr_idx_y) + col]
            } else {
                0f32
            };

        sync_threads();
        for j in 0..TILE_SZ {
            sum += *tile_a.add(thr_idx_y * TILE_SZ + j) * (*tile_b.add(j * TILE_SZ + thr_idx_x));
        }
        sync_threads();
    }
    if row < n && col < n {
        let out_place = &mut *out.add(row * n + col);
        *out_place = sum;
    }
}
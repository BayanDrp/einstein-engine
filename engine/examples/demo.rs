// demo of metric.rs + christoffel.rs
// run from engine/ with: cargo run --example demo

use einstein_engine::geometry::christoffel::christoffel_symbols;
use einstein_engine::geometry::metric::{lower_shift, shift_squared, Metric};
use einstein_engine::geometry::tensor::{Tensor2, Vector};
use einstein_engine::grid::boundary::{BoundaryConfig, BoundaryValues};
use einstein_engine::grid::{Field, Grid};
use std::sync::Arc;

fn main() {
    // a small 3D grid: 4x4x4 points, spacing 0.5
    let grid = Arc::new(Grid::new([4, 4, 4], [0.5, 0.5, 0.5], [0.0, 0.0, 0.0]));

    // lapse: alpha = 1 + 0.1 x, so it varies along x only
    let mut lapse: Field<f64, 3> = Field::zeros(grid.clone());
    grid.for_each_index(|idx| {
        let x = grid.coords(idx)[0];
        lapse.set(idx, 1.0 + 0.1 * x);
    });

    // shift: a small constant x-component, beta^x = 0.05
    let shift: Field<Vector<3>, 3> = Field::new(grid.clone(), Vector::new([0.05, 0.0, 0.0]));

    // spatial 3-metric: flat delta_ij at every point
    let spatial: Field<Tensor2<3>, 3> = Field::new(
        grid.clone(),
        Tensor2::new([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    );

    // --- metric.rs: what live at one grid point --------------------------
    let idx = [2, 2, 2];
    let point = Metric::new(*spatial.get(idx), *lapse.get(idx), *shift.get(idx));
    let lower = lower_shift(&point.spatial, &point.shift);
    println!("at {:?}:", grid.coords(idx));
    println!("  lapse alpha          = {}", point.lapse);
    println!("  shift beta^          = {:?}", point.shift);
    println!("  lowered shift beta_i = {:?}", lower);
    println!("  beta^2               = {}", shift_squared(&point.spatial, &point.shift));
    println!("  4-metric g_ab:");
    for row in point.spacetime().as_4x4() {
        println!("    {:9.4} {:9.4} {:9.4} {:9.4}", row[0], row[1], row[2], row[3]);
    }

    // --- christoffel.rs: symbols over the whole grid ---------------------
    let config = BoundaryConfig::<3>::periodic();
    let values = BoundaryValues::<3>::zeros();
    let gamma_field = christoffel_symbols(&lapse, &shift, &spatial, &config, &values);

    let c = *gamma_field.get(idx);
    println!("Christoffel symbols at {:?}:", grid.coords(idx));
    for a in 0..4 {
        for b in 0..4 {
            for d in 0..4 {
                if c.data[a][b][d] != 0.0 {
                    println!(
                        "  Gamma^{}_{}{} = {:9.4}",
                        a, b, d, c.data[a][b][d]
                    );
                }
            }
        }
    }
}
// Release validation suite for the 3+1 core.
//
// The scenarios live in this directory. This file is only the entry point so
// the suite can run with:
//
// `cargo run --release --example validate`

#[path = "mod.rs"]
mod suite;

fn main() -> std::process::ExitCode {
    suite::run()
}

use std::path::Path;
use std::process::Command;

pub fn run() -> Result<(), String> {
    let engine_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let visualization_dir = engine_dir.join("../visualization");
    let script = engine_dir.join("examples/validate/webgl_test.mjs");
    let node = std::env::var_os("NODE").unwrap_or_else(|| "node".into());

    let output = Command::new(node)
        .arg(&script)
        .current_dir(&visualization_dir)
        .output()
        .map_err(|error| format!("could not start V10 browser test: {error}"))?;

    if output.status.success() {
        print!("{}", String::from_utf8_lossy(&output.stdout));
        eprint!("{}", String::from_utf8_lossy(&output.stderr));
        Ok(())
    } else {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        Err(format!(
            "V10 browser test exited with {}:\n{}{}",
            output.status, stdout, stderr
        ))
    }
}

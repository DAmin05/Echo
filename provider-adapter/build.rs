// protox is a pure-Rust protobuf compiler, so neither local builds nor the
// Docker image need `protoc` installed.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=../proto/provider.proto");
    let fds = protox::compile(["provider.proto"], ["../proto"])?;
    tonic_prost_build::configure()
        .build_client(false)
        .compile_fds(fds)?;
    Ok(())
}

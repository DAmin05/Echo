// protox is a pure-Rust protobuf compiler, so neither local builds nor the
// Docker image need `protoc` installed. The gateway only needs clients.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let protos = ["embedding.proto", "cache.proto", "provider.proto"];
    for p in protos {
        println!("cargo:rerun-if-changed=../proto/{p}");
    }
    let fds = protox::compile(protos, ["../proto"])?;
    tonic_prost_build::configure()
        .build_server(false)
        .compile_fds(fds)?;
    Ok(())
}

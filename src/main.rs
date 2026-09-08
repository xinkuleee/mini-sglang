mod frontend;
#[cfg(feature = "llama")]
mod server;

fn main() {
    if let Err(error) = frontend::run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

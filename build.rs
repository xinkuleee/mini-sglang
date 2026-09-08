//! 默认 core 测试无需 C++、模型或网络。仅 feature=llama 构建固定版本计算桥。
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn run(command: &mut Command) {
    let status = command
        .status()
        .unwrap_or_else(|e| panic!("cannot run {command:?}: {e}"));
    assert!(status.success(), "native build command failed: {command:?}");
}
fn main() {
    for name in [
        "MINISGL_LLAMA_SOURCE",
        "MINISGL_CUDA",
        "MINISGL_NATIVE_JOBS",
        "CMAKE",
        "CMAKE_GENERATOR",
        "CC",
        "CXX",
    ] {
        println!("cargo:rerun-if-env-changed={name}");
    }
    for path in [
        "native/CMakeLists.txt",
        "native/minisgl_bridge.cpp",
        "native/minisgl_bridge.h",
    ] {
        println!("cargo:rerun-if-changed={path}");
    }
    if env::var_os("CARGO_FEATURE_LLAMA").is_none() {
        return;
    }
    let target = env::var("TARGET").expect("TARGET");
    let host = env::var("HOST").expect("HOST");
    assert_eq!(
        target, host,
        "llama feature currently requires a native build (HOST == TARGET)"
    );
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR"));
    let source = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"))
        .join("native");
    let build = out.join("native-build");
    let cmake = env::var_os("CMAKE").unwrap_or_else(|| "cmake".into());
    let mut configure = Command::new(&cmake);
    configure
        .arg("-S")
        .arg(&source)
        .arg("-B")
        .arg(&build)
        .arg("-DCMAKE_BUILD_TYPE=Release")
        .arg("-DMINISGL_BRIDGE_SHARED=ON")
        .arg("-DMINISGL_NATIVE_TESTS=OFF");
    if let Some(generator) = env::var_os("CMAKE_GENERATOR") {
        configure.arg("-G").arg(generator);
    } else if target.contains("windows-gnu") {
        configure.arg("-G").arg("Ninja");
    }
    if let Some(path) = env::var_os("MINISGL_LLAMA_SOURCE") {
        configure.arg(format!(
            "-DMINISGL_LLAMA_SOURCE={}",
            Path::new(&path).display()
        ));
    }
    println!("cargo:rerun-if-env-changed=MINISGL_CPU_BASELINE");
    let baseline = !matches!(
        env::var("MINISGL_CPU_BASELINE").as_deref(),
        Ok("0" | "OFF" | "off" | "false")
    );
    configure.arg(if baseline {
        "-DMINISGL_CPU_BASELINE=ON"
    } else {
        "-DMINISGL_CPU_BASELINE=OFF"
    });
    let cuda = matches!(
        env::var("MINISGL_CUDA").as_deref(),
        Ok("1" | "ON" | "on" | "true")
    );
    configure.arg(if cuda {
        "-DMINISGL_CUDA=ON"
    } else {
        "-DMINISGL_CUDA=OFF"
    });
    run(&mut configure);
    let jobs = env::var("MINISGL_NATIVE_JOBS").unwrap_or_else(|_| {
        std::thread::available_parallelism()
            .map(|n| n.get().min(8).to_string())
            .unwrap_or_else(|_| "2".into())
    });
    run(Command::new(&cmake)
        .arg("--build")
        .arg(&build)
        .arg("--config")
        .arg("Release")
        .arg("--target")
        .arg("minisgl_bridge")
        .arg("--parallel")
        .arg(jobs));
    let libraries = build.join("lib");
    println!("cargo:rustc-link-search=native={}", libraries.display());
    println!("cargo:rustc-link-lib=dylib=minisgl_bridge");
    if target.contains("windows") {
        // Windows 没有 rpath。让 cargo run/test 和直接运行 target/{profile}
        // 二进制都在相邻目录找到唯一 adapter DLL。发行时同样携带此 DLL。
        let profile = out.ancestors().nth(3).expect("Cargo OUT_DIR layout");
        let dll = libraries.join("minisgl_bridge.dll");
        let alternate = libraries.join("libminisgl_bridge.dll");
        let dll = if dll.exists() { dll } else { alternate };
        assert!(
            dll.is_file(),
            "native bridge DLL missing: {}",
            dll.display()
        );
        for destination in [profile.to_path_buf(), profile.join("deps")] {
            fs::create_dir_all(&destination).expect("create native DLL destination");
            fs::copy(
                &dll,
                destination.join(dll.file_name().expect("DLL file name")),
            )
            .unwrap_or_else(|error| {
                panic!("cannot copy native bridge DLL into {}: {error}. Stop running mini-sglang binaries from this target directory before rebuilding", destination.display())
            });
        }
    } else {
        // 开发构建固定到 OUT_DIR；发行包可设置 LD_LIBRARY_PATH/DYLD_LIBRARY_PATH
        // 或把 adapter 放在可执行文件相邻位置。
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", libraries.display());
        if target.contains("apple") {
            println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path");
        } else {
            println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
        }
    }
}

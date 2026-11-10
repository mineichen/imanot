
{
  description = "Deterministic Rust + WASM + Tailwind dev shell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    fenix.url = "github:nix-community/fenix";
    devtools.url = "github:mineichen/nix-devtools";
  };

  outputs = inputs@{ self, nixpkgs, flake-utils, fenix, devtools, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

        rust = with fenix.packages.${system}; combine [
          stable.toolchain
          targets.wasm32-unknown-unknown.stable.rust-std
        ];

        onnxruntime = pkgs.buildEnv {
          name = "onnxruntime-merged";
          paths = [
            pkgs.onnxruntime
            pkgs.onnxruntime.dev
          ];
        };

        envVars = {
          ORT_LIB_PATH = "${onnxruntime}/lib";
          ORT_PREFER_DYNAMIC_LINK = "1";

          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            pkgs.wayland
            pkgs.libxkbcommon
            onnxruntime
            pkgs.stdenv.cc.cc.lib
          ];

          SSL_CERT_FILE =
            "${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt";
        };

        commonBuildInputs = [
          rust
          pkgs.stdenv.cc
          pkgs.trunk
          pkgs.wasm-pack
          pkgs.wayland
          pkgs.mesa
          onnxruntime
          pkgs.pkg-config
        ];

        greet = ''
          echo "===================================="
          echo " Welcome to the deterministic dev shell! "
          echo "===================================="
          rustc --version
          cargo --version
          trunk --version
        '';

        isolated = devtools.lib.isolated {
          inherit pkgs inputs;
          extraMounts = [ "../imbuf" "../imask" ];
          name = "imanot-isolated-dev";
          packages = commonBuildInputs ++ [
            pkgs.ripgrep
            pkgs.git
            pkgs.opencode
            pkgs.busybox
          ];
          claude.pluginDirs = [
            (devtools.lib.claudeRustAnalyzer { inherit pkgs; rust-analyzer = rust; })
          ];
          env = envVars;
          shellHook = greet;
        };

      in
      {
        devShells.default = pkgs.mkShell ({
          buildInputs = commonBuildInputs ++ [
            pkgs.bashInteractive
            pkgs.bash-completion
          ];

          shellHook = greet;
        } // envVars);

        apps.isolated = {
          type = "app";
          program = pkgs.lib.getExe isolated;
        };

        apps.default = {
          type = "app";

          program = "${pkgs.writeShellScriptBin "cursor" ''
            export DISPLAY="''${DISPLAY:-:0}"
            export WAYLAND_DISPLAY="''${WAYLAND_DISPLAY:-wayland-0}"
            export XDG_SESSION_TYPE="''${XDG_SESSION_TYPE:-wayland}"
            export XDG_CURRENT_DESKTOP="''${XDG_CURRENT_DESKTOP:-KDE}"

            exec nix develop . --command \
              ${pkgs.lib.getExe pkgs.code-cursor} \
              --no-sandbox "$PWD"
          ''}/bin/cursor";
        };

        apps.outdated = {
          type = "app";

          program = "${pkgs.writeShellScriptBin "outdated" ''
            exec ${pkgs.cargo-outdated}/bin/cargo-outdated outdated
          ''}/bin/outdated";
        };

        apps.annotationtool-web = {
          type = "app";

          program = "${pkgs.writeShellScriptBin "annotation-tool-web" ''
            set -e

            PROJECT_ROOT="$PWD"

            if [ ! -f "$PROJECT_ROOT/flake.nix" ]; then
              echo "Error: Not in flake root directory" >&2
              exit 1
            fi

            exec nix develop "$PROJECT_ROOT" --command bash -c "
              cd '$PROJECT_ROOT/annotation-tool'
              exec ${pkgs.trunk}/bin/trunk serve --release --no-default-features --features glow
            "
          ''}/bin/annotation-tool-web";
        };

        apps.annotationtool = {
          type = "app";

          program = "${pkgs.writeShellScriptBin "annotation-tool-app" ''
            set -e

            PROJECT_ROOT="$PWD"

            if [ ! -f "$PROJECT_ROOT/flake.nix" ]; then
              echo "Error: Not in flake root directory" >&2
              exit 1
            fi

            # Renderer, optionally the first argument: --glow (default) or --wgpu.
            # wgpu builds into its own target directory, so both builds can
            # exist and run side by side without rebuilding each other.
            case "''${1:-}" in
              --wgpu)
                shift
                FEATURES="--no-default-features --features wayland,sam,wgpu"
                TARGET_DIR="$PROJECT_ROOT/target/wgpu"
                ;;
              --glow)
                shift
                FEATURES="--features sam"
                TARGET_DIR="$PROJECT_ROOT/target"
                ;;
              *)
                FEATURES="--features sam"
                TARGET_DIR="$PROJECT_ROOT/target"
                ;;
            esac

            # Build if needed
            nix develop "$PROJECT_ROOT" --command bash -c \
              "cargo build --release $FEATURES --bin annotation-tool-app --target-dir '$TARGET_DIR'"

            # Run with arguments passed through
            exec nix develop "$PROJECT_ROOT" --command bash -c "
              export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [
                pkgs.libGL
                pkgs.mesa
                pkgs.vulkan-loader
                onnxruntime
                pkgs.stdenv.cc.cc.lib
              ]}:\$LD_LIBRARY_PATH

              export RUST_BACKTRACE=1
              export RUST_LOG=''${RUST_LOG:-imanot=debug}

              if [ \$# -eq 0 ]; then
                exec '$TARGET_DIR/release/annotation-tool-app' ~/Downloads
              else
                exec '$TARGET_DIR/release/annotation-tool-app' \"\$@\"
              fi
            " _ "$@"
          ''}/bin/annotation-tool-app";
        };
      });
}

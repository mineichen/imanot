
{
  description = "Deterministic Rust + WASM + Tailwind dev shell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    fenix.url = "github:nix-community/fenix";
  };

  outputs = { self, nixpkgs, flake-utils, fenix, ... }:
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

        policy = pkgs.writeText "policy.json"
          ''{"default":[{"type":"insecureAcceptAnything"}]}'';

        containername = "imanot-isolated-dev";

        # Make slirp4netns available to the host-side Podman process.
        podmanRuntimePath = pkgs.lib.makeBinPath [
          pkgs.slirp4netns
        ];

        podmanRun = "${pkgs.podman}/bin/podman run --rm -it "
          + "--network=slirp4netns "
          + "--tmpfs /tmp "
          + "-v ../imanot:/workspace/imanot:z "
          + "-v ../imask:/workspace/imask:z "
          + "-v ../imbuf:/workspace/imbuf:z "
          + "-e HOME=/root "
          + "${containername}:latest /bin/entrypoint.sh";

      in
      {
        devShells.default = pkgs.mkShell ({
          buildInputs = commonBuildInputs ++ [
            pkgs.bashInteractive
            pkgs.bash-completion
          ];

          shellHook = greet;
        } // envVars);

        packages.isolated-build = pkgs.dockerTools.buildImage {
          name = containername;
          tag = "latest";

          copyToRoot = pkgs.buildEnv {
            name = containername;

            paths = commonBuildInputs ++ [
              pkgs.bashInteractive
              pkgs.ripgrep
              pkgs.slirp4netns
              pkgs.git
              pkgs.opencode
              pkgs.busybox

              (pkgs.writeScriptBin "entrypoint.sh" ''
                #!${pkgs.bashInteractive}/bin/bash
                ${greet}
                exec ${pkgs.bashInteractive}/bin/bash
              '')
            ];

            pathsToLink = [
              "/bin"
              "/lib"
              "/include"
              "/share"
            ];
          };

          config = {
            Env =
              pkgs.lib.mapAttrsToList
                (k: v: "${k}=${v}")
                envVars
              ++ [ "HOME=/root" ];

            Cmd = [ "/bin/entrypoint.sh" ];

            WorkingDir = "/workspace/imanot";
          };
        };

        apps.isolated-build = {
          type = "app";

          program = toString (pkgs.writeShellScript containername ''
            export PATH="${podmanRuntimePath}:$PATH"

            ${pkgs.podman}/bin/podman rmi ${containername} || true

            ${pkgs.podman}/bin/podman load \
              --signature-policy ${policy} \
              --input ${self.packages.${system}.isolated-build}

            ${podmanRun}
          '');
        };

        apps.isolated-nobuild = {
          type = "app";

          program = toString (pkgs.writeShellScript "run-isolated" ''
            set -euo pipefail

            export PATH="${podmanRuntimePath}:$PATH"

            ${podmanRun}
          '');
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
              exec ${pkgs.trunk}/bin/trunk serve --release --no-default-features
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

            # Build if needed
            nix develop "$PROJECT_ROOT" --command bash -c \
              "cargo build --release --features sam --bin annotation-tool-app"

            # Run with arguments passed through
            exec nix develop "$PROJECT_ROOT" --command bash -c "
              export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [
                pkgs.libGL
                pkgs.mesa
                onnxruntime
                pkgs.stdenv.cc.cc.lib
              ]}:\$LD_LIBRARY_PATH

              export RUST_BACKTRACE=1
              export RUST_LOG=''${RUST_LOG:-imanot=debug}

              if [ \$# -eq 0 ]; then
                exec "$PROJECT_ROOT/target/release/annotation-tool-app" ~/Downloads
              else
                exec "$PROJECT_ROOT/target/release/annotation-tool-app" "\$@"
              fi
            " _ "$@"
          ''}/bin/annotation-tool-app";
        };
      });
}

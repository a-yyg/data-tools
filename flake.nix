{
  description = "hello world application using uv2nix";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";

    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  # outputs =
  #    {
  #      self,
  #      nixpkgs,
  #      uv2nix,
  #      pyproject-nix,
  #      pyproject-build-systems,
  #      ...
  #    }:
  #    let
  #      inherit (nixpkgs) lib;
  #      forAllSystems = lib.genAttrs lib.systems.flakeExposed;

  #      # Load all Python scripts from ./scripts directory
  #      scripts =
  #        lib.mapAttrs
  #          (
  #            name: _:
  #            uv2nix.lib.scripts.loadScript {
  #              script = ./scripts + "/${name}";
  #            }
  #          )
  #          (
  #            lib.filterAttrs (name: type: type == "regular" && lib.hasSuffix ".py" name) (
  #              builtins.readDir ./scripts
  #            )
  #          );

  #      packages' = forAllSystems (
  #        system:
  #        let
  #          pkgs = nixpkgs.legacyPackages.${system};
  #          python = pkgs.python3;
  #          baseSet = pkgs.callPackage pyproject-nix.build.packages {
  #            inherit python;
  #          };
  #        in
  #        lib.mapAttrs (
  #          _name: script:
  #          let
  #            # Create package overlay from script
  #            overlay = script.mkOverlay {
  #              sourcePreference = "wheel";
  #            };

  #            # Construct package set
  #            pythonSet = baseSet.overrideScope (
  #              lib.composeManyExtensions [
  #                pyproject-build-systems.overlays.wheel
  #                overlay
  #              ]
  #            );
  #          in
  #          # Write out an executable script with a shebang pointing to the scripts virtualenv
  #          pkgs.writeScript script.name (
  #            # Returns script as a string with inserted shebang
  #            script.renderScript {
  #              # Construct a virtual environment for script
  #              venv = script.mkVirtualEnv {
  #                inherit pythonSet;
  #              };
  #            }
  #          )
  #        ) scripts
  #      );

  #    in
  #    {
  #      # Drop .py suffix from scripts, making example.py runnable as example
  #      packages = forAllSystems (
  #        system:
  #        lib.mapAttrs' (name: drv: lib.nameValuePair (lib.removeSuffix ".py" name) drv) packages'.${system}
  #      );

  #      # Make each script runnable directly with `nix run`
  #      apps = forAllSystems (
  #        system:
  #        lib.mapAttrs (_name: script: {
  #          type = "app";
  #          program = "${script}";
  #        }) self.packages.${system}
  #      );
  #    };

  outputs =
    {
      nixpkgs,
      pyproject-nix,
      uv2nix,
      pyproject-build-systems,
      ...
    }:
    let
      inherit (nixpkgs) lib;
      forAllSystems = lib.genAttrs lib.systems.flakeExposed;

      workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };

      overlay = workspace.mkPyprojectOverlay {
        sourcePreference = "wheel";
      };

      editableOverlay = workspace.mkEditablePyprojectOverlay {
        root = "$REPO_ROOT";
      };

      pythonSets = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          python = pkgs.python3;
        in
        (pkgs.callPackage pyproject-nix.build.packages {
          inherit python;
        }).overrideScope
          (
            lib.composeManyExtensions [
              pyproject-build-systems.overlays.wheel
              overlay
            ]
          )
      );

    in
    {
      devShells = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          pythonSet = pythonSets.${system}.overrideScope editableOverlay;
          virtualenv = pythonSet.mkVirtualEnv "data-tools-dev-env" workspace.deps.all;
        in
        {
          default = pkgs.mkShell {
            packages = [
              virtualenv
              pkgs.uv
            ];
            env = {
              UV_NO_SYNC = "1";
              UV_PYTHON = pythonSet.python.interpreter;
              UV_PYTHON_DOWNLOADS = "never";
            };
            shellHook = ''
              unset PYTHONPATH
              export REPO_ROOT=$(git rev-parse --show-toplevel)
              # export VENV=${virtualenv}
              # export PYTHONPATH=${virtualenv}/lib/python${lib.versions.majorMinor pythonSet.python.version}/site-packages
            '';
          };
        }
      );

      packages = forAllSystems (system:
        let
          pythonSet = pythonSets.${system};
          pkgs = nixpkgs.legacyPackages.${system};
          inherit (pkgs.callPackages pyproject-nix.build.util { }) mkApplication;
          data-tools = mkApplication {
            venv = pythonSet.mkVirtualEnv "ros2csv-env" workspace.deps.default;
            package = pythonSet.data-tools;
          };
          addMeta =
                    drv:
                    program:
                    drv.overrideAttrs (old: {
                      passthru = lib.recursiveUpdate (old.passthru or { }) {
                        inherit (pythonSet.testing.passthru) tests;
                      };

                      meta = (old.meta or { }) // {
                        mainProgram = program;
                      };
                    });
        in
        {
          venv = pythonSets.${system}.mkVirtualEnv "data-tools-env" workspace.deps.default;
          default = data-tools;
          ros2csv = addMeta data-tools "ros2csv";
          csvgraph = addMeta data-tools "csvgraph";
        });
    };
}

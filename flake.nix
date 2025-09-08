{
  description = "A flake for using Pesto";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  inputs.systems.url = "github:nix-systems/default";
  inputs.flake-utils = {
    url = "github:numtide/flake-utils";
    inputs.systems.follows = "systems";
  };

  outputs =
    { nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs { inherit system; };
        llvm = pkgs.llvmPackages_16;

        ntl = llvm.stdenv.mkDerivation rec {
          pname = "ntl";
          version = "11.5.1";
          src = pkgs.fetchurl {
            url = "https://www.shoup.net/ntl/ntl-${version}.tar.gz";
            sha256 = "sha256-IQ0GwxMGy8bq9oFEU8Vsd22djo3zbXTrMG9qUj0caoo=";
          };
          buildInputs = [
            pkgs.gmp
          ];
          nativeBuildInputs = [
            pkgs.perl # needed for ./configure
          ];
          sourceRoot = "${pname}-${version}/src";

          enableParallelBuilding = true;

          dontAddPrefix = true; # DEF_PREFIX instead
          configurePlatforms = [ ];
          configurePhase = ''
            ./configure PREFIX=$out NATIVE=off NTL_GMP_LIP=on CXX=${llvm.stdenv.cc.targetPrefix}c++
          '';

          doCheck = false;
        };

      in
      {
        devShells.default = pkgs.mkShell.override { stdenv = llvm.stdenv; } {

          packages = with pkgs; [
            git

            llvm.clang-tools
            llvm.stdenv
            llvm.clang
            llvm.libllvm
            llvm.libclang
            llvm.libcxx
            llvm.bintools
            llvm.clang-manpages
            llvm.openmp

            (pkgs.python311.withPackages (
              ppkgs: with ppkgs; [
                numpy
                seaborn
                matplotlib
                pandas
                scipy
              ]
            ))

            libyaml
            gmp
            cmake
            autoconf
            automake
            pkgconf
            libtool
            bison
            flex

            texliveFull

            # trahrhe
            ntl
            maxima
            libmpc

            shfmt
            bc
            cmake-format
            likwid

            reuse

            gitlab-ci-local

          ];
          shellHook = '''';
        };
      }
    );
}

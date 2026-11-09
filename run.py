"""
    python run.py inspect <cfg> [weights]     
    python run.py verify  <cfg> <weights>     
    python run.py prune   <config.yaml>      
"""

from __future__ import annotations

import argparse
import shutil
import sys

from dkprune import config, graph, loop, verify
from dkprune.darknet import DarknetError
from dkprune.io import cfg as cfgio
from dkprune.io.cfg import CfgError
from dkprune.io.weights import WeightsError
from dkprune.layers.base import GraphError



def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dkprune", description="Filter pruning for Darknet/YOLO")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("inspect", help="print table of layers and parameters from cfg (and weights)")
    p.add_argument("cfg")
    p.add_argument("weights", nargs="?")

    p = sub.add_parser("verify", help="verify a pair of cfg and weights")
    p.add_argument("cfg")
    p.add_argument("weights")

    p = sub.add_parser("prune", help="run iterative pruning according to the configuration file")
    p.add_argument("config")

    args = ap.parse_args(argv)
    try:
        if args.command == "inspect":
            print(graph.build(cfgio.load(args.cfg)).table())
            if args.weights:
                print(verify.check_files(args.cfg, args.weights))
        elif args.command == "verify":
            rep = verify.check_files(args.cfg, args.weights)
            print(rep)
            return 0 if rep.ok else 1
        elif args.command == "prune":
            rc = config.load(args.config)
            rc.run_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(args.config, rc.run_dir / "config.yaml")    
            loop.run(rc)
    except (CfgError, GraphError, WeightsError, config.ConfigError, DarknetError, ValueError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
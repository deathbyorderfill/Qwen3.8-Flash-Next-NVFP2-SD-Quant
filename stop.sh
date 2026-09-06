#!/usr/bin/env bash
docker rm -f "${NAME:-sdnvfp2}" >/dev/null 2>&1 && echo stopped || echo "not running"

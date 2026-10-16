#!/bin/bash
# usage: memgate.sh cmd args... : wait until the system reports >= 25% free memory, then exec the command
while [ "$(memory_pressure 2>/dev/null | awk '/free percentage/{gsub("%","",$5); print $5}')" -lt 25 ] 2>/dev/null; do sleep 30; done
exec "$@"

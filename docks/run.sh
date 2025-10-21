#!/bin/bash

DOCKS_PATH=~/.config/docks/docker-compose.yml
docks=$(docker-compose -f $DOCKS_PATH config --services)

usage() {
  /bin/cat << EOM
    Usage:
        $0 [-u] <docks>
    Options:
        -u : update
        -c : cleanup
EOM
exit 0
}

menus() {
  for name in $docks; do
    echo '>' $name
  done
}

update_docker() {
  for name in $docks; do
    docker-compose -f $DOCKS_PATH build $name
  done
  exit 0
}

clean_docker() {
  docker rm $(docker ps -a | grep '^docks/' | awk '{print $3}')
  docker rmi $(docker images -a | grep '^docks/' | awk '{print $3}')
  docker network prune -f
  exit 0
}

while getopts ":uch" opt; do
  case $opt in
    u)
      update_docker
      ;;
    c)
      clean_docker
      ;;
    h | *)
      usage
      ;;
  esac
done
shift $((OPTIND-1))

if [ $# -le 0 ]; then
  menus
else
  docker-compose -f $DOCKS_PATH run --rm -v $(pwd):/workspace $@
fi

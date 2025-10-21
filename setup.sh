#!/bin/bash     
real=`realpath $0`
path=`dirname $real`

function link_file {
    src=$path/$1
    dest=$2
    ln -s -f $src $dest
}

mkdir -p ~/.local/bin
mkdir -p ~/.config/venv

link_file vimrc ~/.vimrc
link_file zshrc ~/.zshrc
link_file tmux.conf ~/.tmux.conf
link_file docks/run.sh ~/.local/bin/docks

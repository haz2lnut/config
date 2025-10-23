## Path
CONFIG_PATH=$HOME/.config
PATH="$HOME/.local/bin:$PATH"

## OMZ
export ZSH="$HOME/.oh-my-zsh"
ZSH_THEME='simple'
plugins=(git tmux tinted-shell)
ZSH_TMUX_AUTOSTART=true
ZSH_TMUX_FIXTERM_WITH_256COLOR=tmux-256color
source $ZSH/oh-my-zsh.sh

## Alias
alias vi='vim'
alias rm='trash'
alias ls="${aliases[ls]} --group-directories-first"
alias pacman='sudo pacman'
alias svi='sudo vim'
alias systemctl='sudo systemctl'
alias journalctl='sudo journalctl'
alias ll='ls -lah'
alias l='ls -lh'
alias python='python3'
cd() { builtin cd "$@"; l; }

update() {
  pacman --noconfirm -Qdtq | ifne sudo pacman -Rns -
  pacman --noconfirm -Syu
  yay -Sua

  vim -E +PluginUpdate +qall

  ~/.tmux/plugins/tpm/bin/update_plugins all

  python ~/.vim/bundle/YouCompleteMe/install.py --clangd-completer --ts-completer --quiet

  omz update
}

mkvenv() {
  python -m venv $CONFIG_PATH/venv/$1
  source $CONFIG_PATH/venv/$1/bin/activate
}

venv() {
  source $CONFIG_PATH/venv/$1/bin/activate
}

x() {
  echo $(($1))
}

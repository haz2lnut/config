"
" Vundle
"
set nocompatible              " be iMproved, required
filetype off                  " required
set rtp+=~/.vim/bundle/Vundle.vim
call vundle#begin()
Plugin 'VundleVim/Vundle.vim'
Plugin 'tinted-theming/tinted-vim'
Plugin 'preservim/nerdtree'
Plugin 'vim-airline/vim-airline'
Plugin 'vim-airline/vim-airline-themes'
Plugin 'tpope/vim-surround'
Plugin 'ycm-core/YouCompleteMe'
Plugin 'sheerun/vim-polyglot'
Plugin 'preservim/tagbar'
call vundle#end()            " required
filetype plugin indent on    " required


"
" Defaults
"
syntax on
set hlsearch
set incsearch
set wildignore=.git,.next,node_modules,__pycache__,.build,.cache,compile_commands.json,*.db,*.o,.eslintrc.json,instance
autocmd FileType * setlocal ts=2 sts=2 sw=2 expandtab smartindent cindent
autocmd FileType markdown setlocal spell spelllang=en,cjk
autocmd FileType c setlocal noexpandtab cc=80
autocmd FileType cpp setlocal noexpandtab cc=80
autocmd FileType make setlocal noexpandtab cc=80
autocmd BufRead,BufNewFIle *.S setlocal filetype=asm
autocmd BufRead,BufNewFIle *.s setlocal filetype=asm
set autowriteall
autocmd BufLeave,BufWinLeave,InsertLeave,CmdlineEnter * if &filetype != 'nerdtree' && &modifiable && filereadable(bufname('%')) | silent! w | endif
set backspace=indent,eol,start
let g:sidebar_width = max([25, winwidth(0) / 5])


"
" Nerdtree
"
let g:NERDTreeQuitOnOpen=1
let g:NERDTreeMinimalUI=1
let g:NERDTreeRespectWildIgnore=1
let g:NERDTreeShowHidden=1
let g:NERDTreeMapOpenSplit=''
let g:NERDTreeMapOpenVSplit=''
let g:NERDTreeMapOpenInTab=''
let g:NERDTreeMapOpenExpl=''
let g:NERDTreeWinSize = g:sidebar_width
" Start NERDTree when Vim is started without file arguments.
autocmd StdinReadPre * let s:std_in=1
autocmd VimEnter * if argc() == 0 && !exists('s:std_in') | NERDTree | endif
" If another buffer tries to replace NERDTree, put it in the other window, and bring back NERDTree.
autocmd BufEnter * if winnr() == winnr('h') && bufname('#') =~ 'NERD_tree_\d\+' && bufname('%') !~ 'NERD_tree_\d\+' && winnr('$') > 1 |
    \ let buf=bufnr() | buffer# | execute "normal! \<C-W>w" | execute 'buffer'.buf | endif


"
" Lists
"
autocmd FileType qf set nobuflisted
let g:quickfix_list_open = 0
let g:location_list_open = 0
let g:location_list_handler = 1
function! OpenList(pfx)
  if a:pfx == 'l'
    try
      lopen
      wincmd J
      8wincmd _
      wincmd p
      let g:location_list_open = 1
      let g:location_list_handler = 1
    catch /E776/
      echohl ErrorMsg
      echo "Location List is Empty."
      echohl None
      return
    endtry
  elseif a:pfx == 'c'
    copen
    wincmd K
    8wincmd _
    wincmd p
    let g:quickfix_list_open = 1
  endif
endfunction

function! ToggleList(pfx)
  if a:pfx == 'l'
    if g:location_list_open
      let g:location_list_open = 0
      let g:location_list_handler = 0
      lclose
    else
      call OpenList(a:pfx)
    endif
  elseif a:pfx == 'c'
    if g:quickfix_list_open
      let g:quickfix_list_open = 0
      cclose
    else
      call OpenList(a:pfx)
    endif
  endif
endfunction

function! CloseBuf()
  cclose
  let g:location_list_open = 0
  lclose
  bdelete
endfunction

autocmd BufWinEnter * if g:quickfix_list_open && &modifiable | call OpenList('c') | endif

function! LocationListHandler()
  if &modifiable && g:location_list_handler
    let l:is_empty = empty(getloclist(0))
    if !l:is_empty
      call OpenList('l')
    else
      let g:location_list_open = 0
      lclose
    endif
  endif
endfunction
autocmd CursorHold,CursorHoldI * call LocationListHandler()

function! CycleList(type, direction)
	if a:type ==# 'c'
		try
			if a:direction ==# 'n'
				cnext
			else
				cprevious
			endif
		catch /E553/
			if a:direction ==# 'n'
				cfirst
			else
				clast
			endif
		catch /E42/
			echohl ErrorMsg
			echo "Quickfix List is Empty."
			echohl None
			return
		endtry
	elseif a:type ==# 'l'
		try
			if a:direction ==# 'n'
				lnext
			else
				lprevious
			endif
		catch /E553/
			if a:direction ==# 'n'
				lfirst
			else
				llast
			endif
		catch /E42/
			echohl ErrorMsg
			echo "Location List is Empty."
			echohl None
			return
		endtry
	endif
endfunction

"
" Colours
"
if filereadable(expand("$HOME/.config/tinted-theming/set_theme.vim"))
	let base16_colorspace=256
	let base16colorspace=256
	let tinted_colorspace=256
	source $HOME/.config/tinted-theming/set_theme.vim
endif


"
" Airline
"
let g:airline#extensions#tabline#enabled = 1
let g:airline#extensions#ycm#enabled = 1
let g:airline_section_y = ''
let g:airline_section_z = '%p%% %l/%L %v'


"
" YCM
"
function! AfterYcm()
  wincmd p
  call OpenList('c')
endfunction
let g:ycm_autoclose_preview_window_after_completion = 1
" Auto location list
let g:ycm_always_populate_location_list = 1


"
" Folding
"
set foldmethod=syntax
autocmd BufWinLeave *.* mkview
autocmd BufWinEnter *.* silent loadview
autocmd FileType python,vim,zsh setlocal foldmethod=indent
" Don't screw up folds when inserting text that might affect them, until
" leaving insert mode. Foldmethod is local to the window. Protect against
" screwing up folding when switching between windows.
autocmd InsertEnter * if !exists('w:last_fdm') | let w:last_fdm=&foldmethod | setlocal foldmethod=manual | endif
autocmd InsertLeave,WinLeave * if exists('w:last_fdm') | let &l:foldmethod=w:last_fdm | unlet w:last_fdm | endif


"
" Tagbar
"
let g:tagbar_width = g:sidebar_width
let g:tagbar_autofocus = 1

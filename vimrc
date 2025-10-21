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


"
" Nerdtree
"
let g:NERDTreeQuitOnOpen=1
let g:NERDTreeMinimalUI=1
let g:NERDTreeRespectWildIgnore=1
let g:NERDTreeShowHidden=1
let g:NERDTreeMapOpenVSplit=''
let g:NERDTreeMapOpenInTab=''
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
" Mapping
"
let mapleader=','
function! Mapmap(lhs, rhs)
  execute 'map '.a:lhs.' '.a:rhs
  execute 'imap '.a:lhs.' <Esc>'.a:rhs
endfunction
" Cursor
noremap s h
noremap S b
noremap t j
noremap T <C-d>
noremap i k
noremap I <C-u>
noremap n l
noremap N w
" Mode
noremap e i
noremap E I
noremap r s
noremap R R
" Operation
noremap m c
noremap M C
noremap c y
noremap C Y
" Motion
noremap k n
noremap K N
" Etc
noremap w :w<CR>
noremap f za
noremap F zR
noremap q @
noremap Q q
noremap j :join!<CR>
noremap J :.-1join!<CR>
noremap y "
noremap Y :registers<CR>
noremap U <C-r>
noremap l :call CycleList('l', 'n')<CR>
noremap L :call CycleList('l', 'p')<CR>
noremap b g;
noremap B g,
noremap <Space> o<Esc>
noremap ( [(
noremap ) ])
noremap < [{
noremap > ]}
noremap ; %
noremap + <C-a>
noremap - <C-x>
noremap h <Plug>(YCMHover)
noremap H :noh<CR>

" Cursor
call Mapmap('<Leader><Left>',  '<C-w>h')
call Mapmap('<Leader><Right>', '<C-w>l')
call Mapmap('<Leader><Up>',    '<C-w>k')
call Mapmap('<Leader><Down>',  '<C-w>j')
call Mapmap('<PageUp>',        '<C-b>')
call Mapmap('<PageDown>',      '<C-f>')
" Window
call Mapmap('<Leader>q', ':wqall<CR>')
call Mapmap('<Leader>C', '<C-w>c')
call Mapmap('<Leader>v', ':vsplit<CR>')
call Mapmap('<Leader>h', ':split<CR>')
" Buffer
call Mapmap('<Leader>c', ':call CloseBuf()<CR>')
call Mapmap('<Leader>e', ':bprevious<CR>')
call Mapmap('<Leader>o', ':bnext<CR>')
" Sizing
call Mapmap('<Leader>+', '<C-w>5+')
call Mapmap('<Leader>-', '<C-w>5-')
call Mapmap('<Leader><', '<C-w>5<')
call Mapmap('<Leader>>', '<C-w>5>')
call Mapmap('<Leader>=', '<C-w>=')
call Mapmap('<Leader>_', '<C-w>_')
call Mapmap('<Leader>\|', '<C-w>\|')
" List
call Mapmap('<Leader>t', ':call ToggleList("l")<CR>')
call Mapmap('<Leader>i', ':call ToggleList("c")<CR>')
call Mapmap('<Leader>l', ':call CycleList("c","n")<CR>')
call Mapmap('<Leader>L', ':call CycleList("c","p")<CR>')
" YCM
call Mapmap('<Leader>p', ':YcmCompleter GoToAlternateFile<CR>')
call Mapmap('<Leader>r', ':YcmCompleter GoToReferences<CR>:call AfterYcm()<CR>')
call Mapmap('<Leader>g', ':YcmCompleter GoTo<CR>')
call Mapmap('<Leader>f', '<Plug>(YCMFindSymbolInWorkspace)')
" Nerdtree
call Mapmap('<Leader>s', ':NERDTreeToggle<CR>')
let g:NERDTreeMenuUp='i'
let g:NERDTreeMenuDown='t'
let g:NERDTreeMapRefresh='s'
let g:NERDTreeMapCustomOpen='n'
" Etc
call Mapmap('<Leader>b', '<C-o>')
call Mapmap('<Leader>B', '<C-i>')

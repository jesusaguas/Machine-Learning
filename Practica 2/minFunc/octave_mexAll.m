% minFunc
fprintf('Compiling minFunc files...\n');
%mex -outdir minFunc/compiled minFunc/mex/mcholC.c
%mex -outdir minFunc/compiled minFunc/mex/lbfgsC.c
%mex -outdir minFunc/compiled minFunc/mex/lbfgsAddC.c
%mex -outdir minFunc/compiled minFunc/mex/lbfgsProdC.c

mex -o minFunc/octave/mcholC.mex      minFunc/mex/mcholC.c
mex -o minFunc/octave/lbfgsC.mex      minFunc/mex/lbfgsC.c
mex -o minFunc/octave/lbfgsAddC.mex   minFunc/mex/lbfgsAddC.c
mex -o minFunc/octave/lbfgsProdC.mex  minFunc/mex/lbfgsProdC.c
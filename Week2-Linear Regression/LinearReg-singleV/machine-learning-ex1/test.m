function test()
	x0 = [1; 2];
	t = linspace (0, 50, 200)';
	x = lsode ("f", x0, t);
	plot (t, x)
endfunction
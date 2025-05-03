.PHONY: run
run:
	python3 app_gc.py --par "a0" &
	python3 app_gc.py --par "a1" &
	python3 app_gc.py --par "lam" &
	python3 app_gc.py --par "G" &
	python3 app_gc.py --par "iota" &
	wait
optimize:
	python3 app_gc_opt.py --par "a0" &
clean:
	killall python3
.PHONY: run
run:
	python3 app_gc.py --par "a0" &
	python3 app_gc.py --par "a1" &
	python3 app_gc.py --par "lam" &
	python3 app_gc.py --par "G" &
	python3 app_gc.py --par "iota" &
	wait
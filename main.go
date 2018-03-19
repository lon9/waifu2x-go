package main

import (
	"github.com/jessevdk/go-flags"
	"github.com/lon9/waifu2x-go/waifu2x"
	"os"
	"runtime"
)

func main() {

	opts := &Options{}
	parser := flags.NewParser(opts, flags.Default)
	parser.Name = "waifu2x-go"
	parser.Usage = "-i[--input] <input-image-path> -o[--output] <output-image-path> -m[--model] <model-path> -c[--cpu] <the-number-of-cpus>"
	_, err := parser.Parse()
	if err != nil {
		os.Exit(1)
	}

	iptImageName := opts.Input
	optImageName := opts.Output
	if optImageName == "" {
		optImageName = "dst.png"
	}
	modelName := opts.ModelName
	numCPU := opts.CPU
	cpus := runtime.NumCPU()
	if numCPU != 0 {
		if numCPU > cpus {
			runtime.GOMAXPROCS(cpus)
		} else {
			runtime.GOMAXPROCS(numCPU)
		}
	}

	w, err := waifu2x.NewWaifu2x(modelName, iptImageName)
	if err != nil {
		panic(err)
	}
	w.Exec()
	if err = w.SaveImage(optImageName); err != nil {
		panic(err)
	}
}

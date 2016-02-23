package main

import (
	"github.com/Rompei/waifu2x-go/waifu2x"
	"github.com/jessevdk/go-flags"
	"os"
)

func main() {

	opts := &Options{}
	parser := flags.NewParser(opts, flags.Default)
	parser.Name = "waifu2x-go"
	parser.Usage = "waifu2x-go -i[--input] <input-image-path> -o[--output] <output-image-path> -m[--model] <model-path>"
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

	w, err := waifu2x.NewWaifu2x(modelName, iptImageName, optImageName)
	if err != nil {
		panic(err)
	}
	w.Exec()
	if err = w.SaveImage(optImageName); err != nil {
		panic(err)
	}
}

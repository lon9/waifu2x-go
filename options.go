package main

// Options is option of the command.
type Options struct {
	Input     string `short:"i" long:"input" description:"Input image file path" required:"true"`
	Output    string `short:"o" long:"output" description:"Output image file path"`
	ModelName string `short:"m" long:"model" description:"Path of model" required:"true"`
	CPU       int    `short:"c" long:"cpu" description:"The number of CPUs used to calcurate"`
}

package waifu2x

/*
This software is waifu2x image reconstructor written in Go.
Model file downloaded from https://marcan.st/transf/scale2.0x_model.json
MIT License https://github.com/nagadomi/waifu2x/blob/master/LICENSE
Reference: https://github.com/nagadomi/waifu2x, https://marcan.st/transf/waifu2x.py
*/

import (
	"encoding/json"
	"fmt"
	"github.com/lon9/mat"
	"github.com/nfnt/resize"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"io/ioutil"
	"math"
	"os"
	"path/filepath"
)

// Model of this program.
type Model struct {
	Weight       [][][][]float32 `json:"weight"`
	NOutputPlane int             `json:"nOutputPlane"`
	KW           int             `json:"kW"`
	KH           int             `json:"kH"`
	Bias         []float32       `json:"bias"`
	NInputPlane  int             `json:"nInputPlane"`
}

// Waifu2x is structure of Waifu2x.
type Waifu2x struct {
	models []Model
	src    image.Image
	dst    *image.RGBA
}

// NewWaifu2x is constructor of Waifu2x.
func NewWaifu2x(modelPath, inputImgPath string) (*Waifu2x, error) {
	var w Waifu2x
	if err := w.loadModel(modelPath); err != nil {
		return nil, err
	}
	if err := w.getImage(inputImgPath); err != nil {
		return nil, err
	}

	return &w, nil
}

func (w *Waifu2x) loadModel(path string) error {

	//Load model from json file.

	f, err := ioutil.ReadFile(path)
	if err != nil {
		return err
	}

	return json.Unmarshal(f, &w.models)
}

func (w *Waifu2x) getImage(path string) error {

	// Getting image from file name.

	sf, err := os.Open(path)
	if err != nil {
		return err
	}

	defer sf.Close()

	img, _, err := image.Decode(sf)
	if err != nil {
		return err
	}
	x := img.Bounds().Max.X
	y := img.Bounds().Max.Y
	w.src = resize.Resize(uint(x*2), uint(y*2), img, resize.NearestNeighbor)
	return nil
}

// SaveImage saves image.
func (w *Waifu2x) SaveImage(name string) error {

	ext := filepath.Ext(name)
	dstFile, err := os.Create(name)
	if err != nil {
		return err
	}
	defer dstFile.Close()
	switch ext {
	case ".png":
		err = png.Encode(dstFile, w.dst)
	case ".jpeg", ".jpg":
		err = jpeg.Encode(dstFile, w.dst, &jpeg.Options{jpeg.DefaultQuality})
	}
	return err
}

func (w *Waifu2x) convertYCbCr(img image.Image) [][]color.YCbCr {

	// Convert color model from RBGA to YCbCr.

	colSize := img.Bounds().Max.X
	rowSize := img.Bounds().Max.Y
	res := make([][]color.YCbCr, rowSize)
	for y := 0; y < rowSize; y++ {
		res[y] = make([]color.YCbCr, colSize)
		for x := 0; x < colSize; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			Y, Cb, Cr := color.RGBToYCbCr(uint8(r), uint8(g), uint8(b))
			res[y][x] = color.YCbCr{Y, Cb, Cr}
		}
	}
	return res
}

func (w *Waifu2x) extY(cl [][]color.YCbCr) [][]float32 {

	// Extract Y value of YCbCr.

	res := make([][]float32, len(cl))
	for i := range cl {
		res[i] = make([]float32, len(cl[i]))
		for j := range cl[i] {
			res[i][j] = float32(cl[i][j].Y)
		}
	}
	return res
}

// Exec execute reconstructing.
func (w *Waifu2x) Exec() {

	// Get Y value.
	c := w.convertYCbCr(w.src)

	width := w.src.Bounds().Max.X
	height := w.src.Bounds().Max.Y
	m := mat.NewMatrix(w.extY(c))

	// Padding.
	padded := m.Pad(uint(len(w.models)), mat.Edge)
	padded = padded.BroadcastDiv(255.0)

	// Prepare planes.
	var planes = []mat.Matrix{*padded}

	// Show progressing.
	progress := 0.0
	count := 0.0
	for _, v := range w.models {
		count += float64(v.NInputPlane * v.NOutputPlane)
	}

	for _, m := range w.models {
		fi := int(math.Min(float64(len(m.Bias)), float64(len(m.Weight))))
		var oPlanes []mat.Matrix
		for i := 0; i < fi; i++ {
			var partial *mat.Matrix
			b := m.Bias[i]
			wgt := m.Weight[i]
			fj := int(math.Min(float64(len(planes)), float64(len(wgt))))
			resCh := make(chan *mat.Matrix, fj)
			for j := 0; j < fj; j++ {
				go func(plane *mat.Matrix, kernel *mat.Matrix, resCh chan *mat.Matrix) {
					m, err := plane.Convolve2d(kernel, 1, 0, mat.Edge)
					if err != nil {
						panic(err)
					}
					resCh <- m
				}(&planes[j], mat.NewMatrix(wgt[j]), resCh)
			}
			for k := 0; k < fj; k++ {
				p := <-resCh
				if partial == nil {
					partial = p
				} else {
					var err error
					partial, err = mat.Add(partial, p)
					if err != nil {
						panic(err)
					}
				}
				progress++
				fmt.Fprintf(os.Stderr, "\r%.1f%%...", 100*progress/count)
			}
			partial = partial.BroadcastAdd(b)
			oPlanes = append(oPlanes, *partial)
		}

		// LeakyReLU
		planes = make([]mat.Matrix, len(oPlanes))
		for i, v := range oPlanes {
			max := v.BroadcastFunc(maximum, float32(0.0))
			min := v.BroadcastFunc(minimum, float32(0.0))
			part := min.BroadcastMul(0.1)
			max, err := mat.Add(max, part)
			if err != nil {
				panic(err)
			}
			planes[i] = *max
		}
	}
	fmt.Println()

	// Assert
	if len(planes) != 1 {
		fmt.Println("error")
		os.Exit(1)
	}

	// Clipping
	//fmt.Println(planes[0])
	res := planes[0].Clip(0.0, 1.0)
	res = res.BroadcastMul(255.0)

	for i := range res.M {
		for j := range res.M[i] {
			c[i][j].Y = uint8(res.M[i][j])
		}
	}

	w.dst = image.NewRGBA(w.src.Bounds())
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			w.dst.Set(x, y, c[y][x])
		}
	}
}

func maximum(a float32, i ...interface{}) float32 {
	arg := i[0].(float32)
	if a > arg {
		return a
	}
	return arg
}

func minimum(a float32, i ...interface{}) float32 {
	arg := i[0].(float32)
	if a < arg {
		return a
	}
	return arg
}

func mul(a, b float32) float32 {
	return a * b
}

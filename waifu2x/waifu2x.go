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
	"github.com/gonum/matrix/mat64"
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
	Weight       [][][][]float64 `json:"weight"`
	NOutputPlane int             `json:"nOutputPlane"`
	KW           int             `json:"kW"`
	KH           int             `json:"kH"`
	Bias         []float64       `json:"bias"`
	NInputPlane  int             `json:"nInputPlane"`
}

// Waifu2x is structure of Waifu2x.
type Waifu2x struct {
	models []Model
	src    image.Image
	dst    *image.RGBA
}

// NewWaifu2x is constructor of Waifu2x.
func NewWaifu2x(modelPath, inputImgPath, outputImagePath string) (*Waifu2x, error) {
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

func (w *Waifu2x) convertYCbCr(img image.Image) []color.YCbCr {

	// Convert color model from RBGA to YCbCr.

	colSize := img.Bounds().Max.X
	rowSize := img.Bounds().Max.Y
	res := make([]color.YCbCr, rowSize*colSize)
	idx := 0
	for y := 0; y < rowSize; y++ {
		for x := 0; x < colSize; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			Y, Cb, Cr := color.RGBToYCbCr(uint8(r), uint8(g), uint8(b))
			res[idx] = color.YCbCr{Y, Cb, Cr}
			idx++
		}
	}
	return res
}

func (w *Waifu2x) extY(cl []color.YCbCr) []float64 {

	// Extract Y value of YCbCr.

	res := make([]float64, len(cl))
	for i, c := range cl {
		res[i] = float64(c.Y)
	}
	return res
}

func (w *Waifu2x) pad(im *mat64.Dense, padding int) *mat64.Dense {

	// Add padding to matrix.

	r, c := im.Dims()
	newRows := r + padding*2
	newCols := c + padding*2
	newVec := make([]float64, newRows*newCols)
	topLeft := im.At(0, 0) / 255.0
	topRight := im.At(0, c-1) / 255.0
	bottomLeft := im.At(r-1, 0) / 255.0
	bottomRight := im.At(r-1, c-1) / 255.0
	idx := 0
	for i := 0; i < padding; i++ {
		for j := 0; j < newCols; j++ {
			if j < padding {
				newVec[idx] = topLeft
				idx++
				continue
			} else if j >= c {
				newVec[idx] = topRight
				idx++
				continue
			}
			newVec[idx] = im.At(padding, j) / 255.0
			idx++
		}
	}
	row := 0
	for i := padding; i < r; i++ {
		for j := 0; j < newCols; j++ {
			if j < padding {
				newVec[idx] = im.At(row, 0) / 255.0
				idx++
				continue
			} else if j >= c {
				newVec[idx] = im.At(row, c-1) / 255.0
				idx++
				continue
			}
			newVec[idx] = im.At(i, j) / 255.0
			idx++
		}
		row++
	}

	for i := r; i < newRows; i++ {
		for j := 0; j < newCols; j++ {
			if j < padding {
				newVec[idx] = bottomLeft
				idx++
				continue
			} else if j >= c {
				newVec[idx] = bottomRight
				idx++
				continue
			}
			newVec[idx] = im.At(r-1, j) / 255.0
			idx++
		}
	}
	return mat64.NewDense(newRows, newCols, newVec)
}

func (w *Waifu2x) calcConv(x, y int, im *mat64.Dense, f [][]float64) float64 {

	// Caluculate convolved value of the point.

	res := 0.0
	res += im.At(x-1, y-1) * f[0][0]
	res += im.At(x, y-1) * f[0][1]
	res += im.At(x+1, y-1) * f[0][2]
	res += im.At(x-1, y) * f[1][0]
	res += im.At(x, y) * f[1][1]
	res += im.At(x+1, y) * f[1][2]
	res += im.At(x-1, y+1) * f[2][0]
	res += im.At(x, y+1) * f[2][1]
	res += im.At(x+1, y+1) * f[2][2]

	return res
}

func (w *Waifu2x) correlate(im *mat64.Dense, f [][]float64) *mat64.Dense {

	// Convolve matrix.

	r, c := im.Dims()
	newRows := r - 2
	newCols := c - 2
	newVec := make([]float64, newRows*newCols)
	idx := 0
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			if i == 0 || j == 0 || i == r-1 || j == c-1 {
				continue
			}
			newVec[idx] = w.calcConv(i, j, im, f)
			idx++
		}
	}

	return mat64.NewDense(newRows, newCols, newVec)
}

func (w *Waifu2x) broadcastMat(mat *mat64.Dense, n float64, f func(a, b float64) float64) *mat64.Dense {

	// Do operation to bitwise of the matrix.

	r, c := mat.Dims()
	newVec := make([]float64, r*c)
	idx := 0
	for y := 0; y < r; y++ {
		for x := 0; x < c; x++ {
			newVec[idx] = f(mat.At(y, x), n)
			idx++
		}
	}
	return mat64.NewDense(r, c, newVec)
}

func (w *Waifu2x) broadcastVec(vec []float64, n float64, f func(a, b float64) float64) {
	for i, v := range vec {
		vec[i] = f(v, n)
	}
}

func (w *Waifu2x) clip(im *mat64.Dense, start, end float64) []float64 {
	r, c := im.Dims()
	newVec := make([]float64, r*c)
	idx := 0
	for y := 0; y < r; y++ {
		for x := 0; x < c; x++ {
			e := im.At(y, x)
			if e < start {
				newVec[idx] = start
			} else if e > end {
				newVec[idx] = end
			} else {
				newVec[idx] = e
			}
			idx++
		}
	}
	return newVec
}

// Exec execute reconstructing.
func (w *Waifu2x) Exec() {

	// Get Y value.
	c := w.convertYCbCr(w.src)

	width := w.src.Bounds().Max.X
	height := w.src.Bounds().Max.Y
	m := mat64.NewDense(height, width, w.extY(c))

	// Padding.
	padded := w.pad(m, len(w.models))

	// Prepare planes.
	var planes = []mat64.Dense{*padded}

	// Show progressing.
	progress := 0.0
	count := 0.0
	for _, v := range w.models {
		count += float64(v.NInputPlane * v.NOutputPlane)
	}

	for _, m := range w.models {
		fi := int(math.Min(float64(len(m.Bias)), float64(len(m.Weight))))
		var oPlanes []mat64.Dense
		for i := 0; i < fi; i++ {
			var partial *mat64.Dense
			b := m.Bias[i]
			wgt := m.Weight[i]
			fj := int(math.Min(float64(len(planes)), float64(len(wgt))))
			resCh := make(chan *mat64.Dense, fj)
			for j := 0; j < fj; j++ {
				go func(plane *mat64.Dense, kernel [][]float64, resCh chan *mat64.Dense) {
					resCh <- w.correlate(plane, kernel)
				}(&planes[j], wgt[j], resCh)
			}
			for k := 0; k < fj; k++ {
				p := <-resCh
				if partial == nil {
					partial = p
				} else {
					partial.Add(partial, p)
				}
				progress++
				fmt.Fprintf(os.Stderr, "\r%.1f%%...", 100*progress/count)
			}
			partial = w.broadcastMat(partial, b, add)
			oPlanes = append(oPlanes, *partial)
		}
		planes = make([]mat64.Dense, len(oPlanes))
		for i, v := range oPlanes {
			max := w.broadcastMat(&v, 0, maximum)
			min := w.broadcastMat(&v, 0, minimum)
			part := w.broadcastMat(min, 0.1, mul)
			max.Add(max, part)
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
	vec := w.clip(&planes[0], 0.0, 1.0)
	w.broadcastVec(vec, 255.0, mul)

	for i, v := range vec {
		c[i].Y = uint8(v)
	}

	idx := 0
	w.dst = image.NewRGBA(w.src.Bounds())
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			w.dst.Set(x, y, c[idx])
			idx++
		}
	}
}

/* Helper functions */

func mul(a, b float64) float64 {
	return a * b
}

func div(a, b float64) float64 {
	return a / b
}

func add(a, b float64) float64 {
	return a + b
}

func maximum(a, b float64) float64 {
	if a >= b {
		return a
	}
	return b
}
func minimum(a, b float64) float64 {
	if a <= b {
		return a
	}
	return b
}

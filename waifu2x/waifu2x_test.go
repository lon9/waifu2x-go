package waifu2x

import (
	"testing"
)

func TestWaifu2x(t *testing.T) {
	w, err := NewWaifu2x("/export/space/takaha-r/waifu2x/models/anime_style_art/scale2.0x_model.json", "miku_small.png")
	if err != nil {
		t.Log("Cant Initialize")
		t.Fatal(err)
	}
	w.Exec()
	if err = w.SaveImage("miku.png"); err != nil {
		t.Fatal(err)
	}
}

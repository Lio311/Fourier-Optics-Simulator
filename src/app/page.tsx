"use client";

import { useState, useEffect, useRef } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Play, RotateCcw, Upload, Image as ImageIcon, Waves, Aperture } from "lucide-react";
import gsap from "gsap";
import FFT from "fft.js";

// Utility to convert color to grayscale
const rgbaToGrayscale = (data: Uint8ClampedArray, width: number, height: number): Float32Array => {
  const gray = new Float32Array(width * height);
  for (let i = 0; i < width * height; i++) {
    const r = data[i * 4];
    const g = data[i * 4 + 1];
    const b = data[i * 4 + 2];
    gray[i] = 0.299 * r + 0.587 * g + 0.114 * b;
  }
  return gray;
};

// Map grayscale to ImageData
const mapToImageData = (gray: Float32Array, data: Uint8ClampedArray, width: number, height: number) => {
  let min = Infinity, max = -Infinity;
  for (let i = 0; i < width * height; i++) {
    if (gray[i] < min) min = gray[i];
    if (gray[i] > max) max = gray[i];
  }
  const range = max - min || 1;
  for (let i = 0; i < width * height; i++) {
    const val = ((gray[i] - min) / range) * 255;
    data[i * 4] = val;
    data[i * 4 + 1] = val;
    data[i * 4 + 2] = val;
    data[i * 4 + 3] = 255;
  }
};

// 2D FFT
const fft2D = (inputReal: Float32Array, width: number, height: number) => {
  const fftW = new FFT(width);
  const fftH = new FFT(height);
  const complexData = new Float32Array(width * height * 2);

  // Rows
  for (let y = 0; y < height; y++) {
    const rowReal = new Float32Array(width);
    for (let x = 0; x < width; x++) rowReal[x] = inputReal[y * width + x] * ( (x+y)%2 === 0 ? 1 : -1 ); // Shift to center
    const outComplex = new Float32Array(width * 2);
    fftW.realTransform(outComplex, rowReal);
    for (let x = 0; x < width; x++) {
      complexData[(y * width + x) * 2] = outComplex[x * 2];
      complexData[(y * width + x) * 2 + 1] = outComplex[x * 2 + 1];
    }
  }

  // Columns
  for (let x = 0; x < width; x++) {
    const colComplex = new Float32Array(height * 2);
    for (let y = 0; y < height; y++) {
      colComplex[y * 2] = complexData[(y * width + x) * 2];
      colComplex[y * 2 + 1] = complexData[(y * width + x) * 2 + 1];
    }
    const outComplex = new Float32Array(height * 2);
    fftH.transform(outComplex, colComplex);
    for (let y = 0; y < height; y++) {
      complexData[(y * width + x) * 2] = outComplex[y * 2];
      complexData[(y * width + x) * 2 + 1] = outComplex[y * 2 + 1];
    }
  }

  return complexData;
};

// 2D IFFT
const ifft2D = (inputComplex: Float32Array, width: number, height: number) => {
  const fftW = new FFT(width);
  const fftH = new FFT(height);
  const outComplex = new Float32Array(width * height * 2);

  // Columns
  for (let x = 0; x < width; x++) {
    const colComplex = new Float32Array(height * 2);
    for (let y = 0; y < height; y++) {
      colComplex[y * 2] = inputComplex[(y * width + x) * 2];
      colComplex[y * 2 + 1] = inputComplex[(y * width + x) * 2 + 1];
    }
    const colOut = new Float32Array(height * 2);
    fftH.inverseTransform(colOut, colComplex);
    for (let y = 0; y < height; y++) {
      outComplex[(y * width + x) * 2] = colOut[y * 2];
      outComplex[(y * width + x) * 2 + 1] = colOut[y * 2 + 1];
    }
  }

  // Rows
  const finalReal = new Float32Array(width * height);
  for (let y = 0; y < height; y++) {
    const rowComplex = new Float32Array(width * 2);
    for (let x = 0; x < width; x++) {
      rowComplex[x * 2] = outComplex[(y * width + x) * 2];
      rowComplex[x * 2 + 1] = outComplex[(y * width + x) * 2 + 1];
    }
    const rowOut = new Float32Array(width * 2);
    fftW.inverseTransform(rowOut, rowComplex);
    for (let x = 0; x < width; x++) {
      finalReal[y * width + x] = Math.abs(rowOut[x * 2]) * ( (x+y)%2 === 0 ? 1 : -1 ); // Shift back
    }
  }

  return finalReal;
};

// Draw Magnitude Spectrum
const drawMagnitude = (complex: Float32Array, width: number, height: number, data: Uint8ClampedArray) => {
  let min = Infinity, max = -Infinity;
  const mag = new Float32Array(width * height);
  for (let i = 0; i < width * height; i++) {
    const re = complex[i * 2];
    const im = complex[i * 2 + 1];
    let val = Math.log(1 + Math.sqrt(re * re + im * im));
    mag[i] = val;
    if (val < min) min = val;
    if (val > max) max = val;
  }
  const range = max - min || 1;
  for (let i = 0; i < width * height; i++) {
    const val = ((mag[i] - min) / range) * 255;
    data[i * 4] = val;
    data[i * 4 + 1] = val;
    data[i * 4 + 2] = val;
    data[i * 4 + 3] = 255;
  }
};

export default function FourierSimulator() {
  const containerRef = useRef<HTMLDivElement>(null);
  
  // Canvases
  const originalCanvas = useRef<HTMLCanvasElement>(null);
  const fourierCanvas = useRef<HTMLCanvasElement>(null);
  const resultCanvas = useRef<HTMLCanvasElement>(null);

  const [imageSize] = useState(256); // Keep it power of 2
  const [filterType, setFilterType] = useState("lowpass");
  const [radius, setRadius] = useState<number>(30);
  const [isProcessing, setIsProcessing] = useState(false);
  const [originalGray, setOriginalGray] = useState<Float32Array | null>(null);

  useEffect(() => {
    const ctx = gsap.context(() => {
      gsap.from(".anim-header", { y: -30, opacity: 0, duration: 0.8, ease: "power3.out" });
      gsap.from(".anim-card", { scale: 0.95, opacity: 0, duration: 0.8, stagger: 0.1, ease: "power3.out", delay: 0.2 });
    }, containerRef);
    
    // Default synthetic image
    drawDefaultImage();

    return () => ctx.revert();
  }, []);

  const drawDefaultImage = () => {
    if (!originalCanvas.current) return;
    const ctx = originalCanvas.current.getContext("2d");
    if (!ctx) return;

    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, imageSize, imageSize);
    
    // Draw some patterns
    ctx.fillStyle = "white";
    ctx.fillRect(100, 100, 56, 56);
    
    ctx.strokeStyle = "white";
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.arc(60, 60, 30, 0, 2 * Math.PI);
    ctx.stroke();

    for (let i = 0; i < 256; i+=10) {
        ctx.beginPath();
        ctx.moveTo(10, i);
        ctx.lineTo(40, i);
        ctx.stroke();
    }

    const imgData = ctx.getImageData(0, 0, imageSize, imageSize);
    setOriginalGray(rgbaToGrayscale(imgData.data, imageSize, imageSize));
  };

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      const img = new Image();
      img.onload = () => {
        if (!originalCanvas.current) return;
        const ctx = originalCanvas.current.getContext("2d");
        if (!ctx) return;
        
        ctx.drawImage(img, 0, 0, imageSize, imageSize);
        const imgData = ctx.getImageData(0, 0, imageSize, imageSize);
        setOriginalGray(rgbaToGrayscale(imgData.data, imageSize, imageSize));
      };
      img.src = event.target?.result as string;
    };
    reader.readAsDataURL(file);
  };

  const applyFourier = () => {
    if (!originalGray || !fourierCanvas.current || !resultCanvas.current) return;
    setIsProcessing(true);

    setTimeout(() => {
      const w = imageSize;
      const h = imageSize;

      // 1. Transform
      const fourierComplex = fft2D(originalGray, w, h);

      // 2. Draw magnitude
      const fCtx = fourierCanvas.current!.getContext("2d")!;
      const fData = fCtx.createImageData(w, h);
      drawMagnitude(fourierComplex, w, h, fData.data);
      
      // 3. Apply Filter (Masking)
      const cx = w / 2;
      const cy = h / 2;
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const dist = Math.sqrt((x - cx)**2 + (y - cy)**2);
          let keep = 1;
          
          if (filterType === "lowpass") {
            keep = dist <= radius ? 1 : 0;
          } else if (filterType === "highpass") {
            keep = dist >= radius ? 1 : 0;
          } else if (filterType === "bandpass") {
            keep = (dist >= radius - 10 && dist <= radius + 10) ? 1 : 0;
          }

          if (keep === 0) {
            fourierComplex[(y * w + x) * 2] = 0;
            fourierComplex[(y * w + x) * 2 + 1] = 0;
            // Draw red overlay on filtered parts in fourier canvas
            const idx = (y * w + x) * 4;
            fData.data[idx] = Math.max(0, fData.data[idx] - 50); // R
            fData.data[idx+1] = Math.min(255, fData.data[idx+1] + 100); // G (show filter as a hue)
          }
        }
      }
      fCtx.putImageData(fData, 0, 0);

      // 4. Inverse Transform
      const filteredReal = ifft2D(fourierComplex, w, h);

      // 5. Draw Result
      const rCtx = resultCanvas.current!.getContext("2d")!;
      const rData = rCtx.createImageData(w, h);
      mapToImageData(filteredReal, rData.data, w, h);
      rCtx.putImageData(rData, 0, 0);

      setIsProcessing(false);
    }, 100); // small timeout to allow UI update
  };

  const handleSliderChange = (val: number | readonly number[]) => {
      const v = Array.isArray(val) ? val[0] : val;
      setRadius(v);
  };

  return (
    <div className="relative min-h-screen text-slate-100 p-4 md:p-8 overflow-hidden" dir="rtl" ref={containerRef}>
      <div className="bg-animation"></div>

      <div className="container max-w-6xl mx-auto relative z-10 space-y-8">
        <header className="anim-header flex flex-col md:flex-row justify-between items-center pb-6 border-b border-slate-700/50">
          <div className="flex items-center gap-3">
            <Aperture className="w-10 h-10 text-cyan-400" />
            <div>
              <h1 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-purple-400 to-cyan-400">
                סימולטור אופטיקה - מישור פורייה
              </h1>
              <p className="text-sm text-slate-400 mt-1">מערכת 4f, התמרות מרחביות וסינון תדרים</p>
            </div>
          </div>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Controls */}
          <Card className="anim-card glass-card lg:col-span-1 border-slate-700/50">
            <CardHeader>
              <CardTitle className="text-lg text-white flex items-center gap-2">
                <Waves className="w-5 h-5 text-purple-400" /> בקרה וסינון
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              
              <div className="space-y-3">
                <label className="text-sm text-slate-200 font-medium flex items-center gap-2">
                  <Upload className="w-4 h-4" /> טען תמונה
                </label>
                <Input 
                  type="file" 
                  accept="image/*" 
                  onChange={handleImageUpload} 
                  className="bg-slate-800/50 border-slate-600 text-slate-200 cursor-pointer"
                />
              </div>

              <div className="space-y-3">
                <label className="text-sm text-slate-200 font-medium">סוג מסנן (Filter)</label>
                <Select value={filterType} onValueChange={(val) => { if (val) setFilterType(val); }}>
                  <SelectTrigger className="bg-slate-800/50 border-slate-600">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-slate-800 border-slate-600 text-slate-200">
                    <SelectItem value="lowpass">Low-pass (מעביר תדרים נמוכים)</SelectItem>
                    <SelectItem value="highpass">High-pass (מעביר תדרים גבוהים)</SelectItem>
                    <SelectItem value="bandpass">Band-pass (מעביר תדרי ביניים)</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-4">
                <div className="flex justify-between">
                  <label className="text-sm text-slate-200 font-medium">רדיוס החיתוך (Cutoff)</label>
                  <span className="text-cyan-400 font-mono text-sm">{radius}px</span>
                </div>
                <Slider 
                  value={[radius]} 
                  onValueChange={handleSliderChange} 
                  max={128} 
                  step={1}
                  className="[&_[role=slider]]:bg-cyan-400 [&_[role=slider]]:border-cyan-400"
                />
              </div>

              <div className="pt-4 border-t border-slate-700/50 flex flex-col gap-3">
                <Button 
                  onClick={applyFourier} 
                  disabled={isProcessing}
                  className="w-full bg-gradient-to-r from-purple-600 to-cyan-600 hover:from-purple-500 hover:to-cyan-500 text-white font-semibold"
                >
                  <Play className="w-4 h-4 ml-2" />
                  {isProcessing ? "מעבד..." : "הפעל התמרה"}
                </Button>
                
                <Button 
                  variant="outline"
                  onClick={drawDefaultImage} 
                  className="w-full border-slate-600 text-slate-200 hover:bg-slate-800"
                >
                  <RotateCcw className="w-4 h-4 ml-2" />
                  אפס תמונה
                </Button>
              </div>
            </CardContent>
          </Card>

          {/* Visualizations */}
          <div className="lg:col-span-3 grid grid-cols-1 md:grid-cols-3 gap-6">
            
            <Card className="anim-card glass-card border-slate-700/50 overflow-hidden flex flex-col">
              <CardHeader className="bg-slate-900/40 border-b border-slate-700/50 py-3">
                <CardTitle className="text-sm text-center text-slate-200 font-medium">תמונת מקור (Input)</CardTitle>
              </CardHeader>
              <CardContent className="p-0 flex-1 flex items-center justify-center bg-black/40 min-h-[256px]">
                <canvas 
                  ref={originalCanvas} 
                  width={imageSize} 
                  height={imageSize} 
                  className="max-w-full h-auto shadow-2xl"
                />
              </CardContent>
            </Card>

            <Card className="anim-card glass-card border-slate-700/50 overflow-hidden flex flex-col relative">
              <CardHeader className="bg-slate-900/40 border-b border-slate-700/50 py-3">
                <CardTitle className="text-sm text-center text-purple-300 font-medium">מישור פורייה (Frequency)</CardTitle>
              </CardHeader>
              <CardContent className="p-0 flex-1 flex items-center justify-center bg-black/60 min-h-[256px] relative">
                <canvas 
                  ref={fourierCanvas} 
                  width={imageSize} 
                  height={imageSize} 
                  className="max-w-full h-auto shadow-2xl"
                />
                {isProcessing && (
                  <div className="absolute inset-0 bg-slate-900/50 flex items-center justify-center backdrop-blur-sm">
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-purple-500"></div>
                  </div>
                )}
              </CardContent>
            </Card>

            <Card className="anim-card glass-card border-slate-700/50 overflow-hidden flex flex-col">
              <CardHeader className="bg-slate-900/40 border-b border-slate-700/50 py-3">
                <CardTitle className="text-sm text-center text-cyan-300 font-medium">תוצאה מסוננת (Output)</CardTitle>
              </CardHeader>
              <CardContent className="p-0 flex-1 flex items-center justify-center bg-black/40 min-h-[256px]">
                <canvas 
                  ref={resultCanvas} 
                  width={imageSize} 
                  height={imageSize} 
                  className="max-w-full h-auto shadow-2xl"
                />
              </CardContent>
            </Card>
            
          </div>
        </div>
      </div>
    </div>
  );
}


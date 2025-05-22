"use client"

import { useCallback } from "react"
import { useDropzone } from "react-dropzone"
import { X, Upload, Eye } from "lucide-react"
import { Button } from "@/components/ui/button"

interface ImageUploaderProps {
  onFilesSelected: (files: File[]) => void
  onPreview: (file: File) => void
  files: File[]
}

export function ImageUploader({ onFilesSelected, onPreview, files }: ImageUploaderProps) {
  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      onFilesSelected([...files, ...acceptedFiles])
    },
    [onFilesSelected, files],
  )

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "image/*": [".jpeg", ".jpg", ".png"],
    },
    multiple: true,
  })

  const removeFile = (index: number) => {
    const newFiles = [...files]
    newFiles.splice(index, 1)
    onFilesSelected(newFiles)
  }

  return (
    <div className="space-y-4">
      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors
          ${isDragActive ? "border-primary bg-primary/5" : "border-gray-300 hover:border-primary/50"}`}
      >
        <input {...getInputProps()} />
        <div className="flex flex-col items-center justify-center space-y-2">
          <Upload className="h-8 w-8 text-muted-foreground" />
          <p className="text-sm font-medium">
            {isDragActive ? "Thả hình ảnh ở đây" : "Kéo và thả hình ảnh hoặc nhấp để chọn"}
          </p>
          <p className="text-xs text-muted-foreground">Hỗ trợ: JPG, JPEG, PNG</p>
        </div>
      </div>

      {files.length > 0 && (
        <div>
          <div className="flex justify-between items-center mb-2">
            <h3 className="text-sm font-medium">Hình ảnh đã chọn ({files.length})</h3>
            <Button variant="ghost" size="sm" onClick={() => onFilesSelected([])} className="h-8 text-xs">
              Xóa tất cả
            </Button>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
            {files.map((file, index) => (
              <div key={index} className="relative group">
                <div className="aspect-square rounded-md overflow-hidden border bg-muted">
                  <img
                    src={URL.createObjectURL(file) || "/placeholder.svg"}
                    alt={`Preview ${index}`}
                    className="h-full w-full object-cover"
                  />
                </div>
                <div className="absolute top-1 right-1 flex gap-1">
                  <Button
                    variant="secondary"
                    size="icon"
                    className="h-6 w-6 bg-white/80 hover:bg-white opacity-0 group-hover:opacity-100 transition-opacity"
                    onClick={() => onPreview(file)}
                  >
                    <Eye className="h-3 w-3" />
                    <span className="sr-only">Xem</span>
                  </Button>
                  <Button
                    variant="destructive"
                    size="icon"
                    className="h-6 w-6 opacity-0 group-hover:opacity-100 transition-opacity"
                    onClick={() => removeFile(index)}
                  >
                    <X className="h-3 w-3" />
                    <span className="sr-only">Xóa</span>
                  </Button>
                </div>
                <p className="text-xs truncate mt-1">{file.name}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

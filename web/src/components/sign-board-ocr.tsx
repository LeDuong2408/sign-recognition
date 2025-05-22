"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
// import { uploadAndExtract, getUploadHistory } from "@/app/actions"
import { ImageUploader } from "@/components/image-uploader"
import { ResultsDisplay } from "@/components/results-display"
import { HistorySidebar } from "@/components/history-sidebar"
import { LoadingOverlay } from "@/components/loading-overlay"
import { ImagePreview } from "@/components/image-preview"
import { Loader2 } from "lucide-react"
import { useToast } from "@/components/ui/use-toast"
import { StoreData } from "@/types/store-data"
import { getUploadHistory, HistoryItem, uploadAndExtract } from "@/hooks/actions"

export function SignBoardOCR() {
  const [files, setFiles] = useState<File[]>([])
  const [results, setResults] = useState<StoreData[]>([])
  const [isProcessing, setIsProcessing] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [sampleImages, setSampleImages] = useState<HistoryItem[]>([])
  const [loadingMessage, setLoadingMessage] = useState("Đang xử lý...")
  const [previewImage, setPreviewImage] = useState<{
    url: string
    title?: string
    storeData?: StoreData
  } | null>(null)
  const { toast } = useToast()

  useEffect(() => {
    // Lấy một số hình ảnh mẫu khi component được mount
    const fetchSampleImages = async () => {
      try {
        const historyData = await getUploadHistory()
        // Chỉ lấy tối đa 6 hình ảnh mẫu
        setSampleImages(historyData.slice(0, 6))
      } catch (err) {
        console.error("Lỗi khi lấy hình ảnh mẫu:", err)
        toast({
          title: "Không thể tải hình ảnh mẫu",
          description: "Đã xảy ra lỗi khi tải hình ảnh mẫu",
          variant: "destructive",
        })
      }
    }

    fetchSampleImages()
  }, [toast])

  const handleFilesSelected = (selectedFiles: File[]) => {
    setFiles(selectedFiles)
    setResults([])
    setError(null)
  }

  const handleSampleSelect = (url: string, name: string) => {
    // Tạo một File object từ URL hình ảnh mẫu
    fetch(url)
      .then((res) => res.blob())
      .then((blob) => {
        const file = new File([blob], name, { type: blob.type })
        setFiles((prevFiles) => [...prevFiles, file])
        setResults([])
        setError(null)
      })
      .catch((err) => {
        console.error("Lỗi khi tải hình ảnh mẫu:", err)
        toast({
          title: "Không thể tải hình ảnh mẫu",
          description: "Đã xảy ra lỗi khi tải hình ảnh mẫu",
          variant: "destructive",
        })
      })
  }

  const handlePreviewSample = (url: string, name: string) => {
    setPreviewImage({ url, title: name })
  }

  const handlePreviewFile = (file: File) => {
    const url = URL.createObjectURL(file)
    setPreviewImage({ url, title: file.name })
  }

  const handlePreviewResult = (url: string, title: string, storeData: StoreData) => {
    setPreviewImage({ url, title, storeData })
  }

  const handleExtract = async () => {
    if (files.length === 0) {
      setError("Vui lòng tải lên ít nhất một hình ảnh")
      return
    }

    setIsProcessing(true)
    setError(null)

    try {
      // Cập nhật thông báo loading theo tiến trình
      setLoadingMessage("Đang tải hình ảnh lên máy chủ...")

      const formData = new FormData()
      files.forEach((file, index) => {
        formData.append(`image-${index}`, file)
      })

      // Đợi một chút để người dùng thấy thông báo đầu tiên
      await new Promise((resolve) => setTimeout(resolve, 800))
      setLoadingMessage("Đang trích xuất dữ liệu từ hình ảnh...")

      const { results: extractedData } = await uploadAndExtract(formData)
      setResults(extractedData)

      toast({
        title: "Trích xuất thành công",
        description: `Đã trích xuất dữ liệu từ ${files.length} hình ảnh`,
      })
    } catch (err) {
      console.error("Lỗi khi xử lý hình ảnh:", err)
      setError("Đã xảy ra lỗi khi xử lý hình ảnh. Vui lòng thử lại.")
      toast({
        title: "Trích xuất thất bại",
        description: "Đã xảy ra lỗi khi xử lý hình ảnh",
        variant: "destructive",
      })
    } finally {
      setIsProcessing(false)
    }
  }

  return (
    <>
      <LoadingOverlay isLoading={isProcessing} message={loadingMessage} />

      <ImagePreview
        isOpen={!!previewImage}
        imageUrl={previewImage?.url || ""}
        imageTitle={previewImage?.title}
        storeData={previewImage?.storeData}
        onClose={() => setPreviewImage(null)}
      />

      <div className="w-full max-w-7xl">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Sidebar hình ảnh mẫu */}
          <div className="lg:col-span-1 h-fit">
            <HistorySidebar
              sampleImages={sampleImages}
              onSelect={handleSampleSelect}
              onPreview={handlePreviewSample}
              isLoading={isProcessing}
            />
          </div>

          {/* Khu vực chính */}
          <div className="lg:col-span-3 space-y-6">
            <Card>
              <CardContent className="pt-6">
                <ImageUploader onFilesSelected={handleFilesSelected} files={files} onPreview={handlePreviewFile} />

                {error && <div className="mt-4 text-red-500 text-sm">{error}</div>}

                <div className="mt-6 flex justify-center">
                  <Button
                    onClick={handleExtract}
                    disabled={isProcessing || files.length === 0}
                    className="w-full md:w-auto"
                  >
                    {isProcessing ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        Đang trích xuất...
                      </>
                    ) : (
                      "Trích xuất dữ liệu"
                    )}
                  </Button>
                </div>
              </CardContent>
            </Card>

            {results.length > 0 && (
              <ResultsDisplay
                results={results}
                images={files.map((file) => ({
                  name: file.name,
                  url: URL.createObjectURL(file),
                }))}
                onPreviewImage={handlePreviewResult}
              />
            )}
          </div>
        </div>
      </div>
    </>
  )
}

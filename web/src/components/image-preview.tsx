"use client"

import type React from "react"
import type { StoreData } from "@/types/store-data"

import { useState, useEffect, useRef } from "react"
import { X, ZoomIn, ZoomOut, Building2, MapPin, Phone } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"

interface ImagePreviewProps {
  isOpen: boolean
  imageUrl: string
  onClose: () => void
  imageTitle?: string
  storeData?: StoreData
}

export function ImagePreview({ isOpen, imageUrl, onClose, imageTitle, storeData }: ImagePreviewProps) {
  const [scale, setScale] = useState(1)
  const [showInfo, setShowInfo] = useState(!!storeData)
  const [position, setPosition] = useState({ x: 0, y: 0 })
  const [isDragging, setIsDragging] = useState(false)
  const dragStartRef = useRef({ x: 0, y: 0 })
  const imageRef = useRef<HTMLImageElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === "Escape" && isOpen) {
        onClose()
      }
    }

    window.addEventListener("keydown", handleEscape)
    return () => window.removeEventListener("keydown", handleEscape)
  }, [isOpen, onClose])

  // Ngăn chặn cuộn trang khi modal mở
  useEffect(() => {
    if (isOpen) {
      const originalStyle = window.getComputedStyle(document.body).overflow
      document.body.style.overflow = "hidden"
      return () => {
        document.body.style.overflow = originalStyle
      }
    }
    return undefined;
  }, [isOpen])

  

  // Xử lý sự kiện wheel toàn cục
  useEffect(() => {
    const handleGlobalWheel = (e: WheelEvent) => {
      if (isOpen && containerRef.current?.contains(e.target as Node)) {
        e.preventDefault()
        e.stopPropagation()

        const delta = e.deltaY * -0.01
        const newScale = Math.min(Math.max(scale + delta, 1), 3)
        setScale(newScale)

        if (newScale === 1) {
          setPosition({ x: 0, y: 0 })
        }

        return false
      }
      return false;
    }

    // Thêm passive: false để có thể gọi preventDefault()
    window.addEventListener("wheel", handleGlobalWheel, { passive: false })
    return () => window.removeEventListener("wheel", handleGlobalWheel)
  }, [isOpen, scale])

  useEffect(() => {
    // Reset scale and position when opening a new image
    if (isOpen) {
      setScale(1)
      setShowInfo(!!storeData)
      setPosition({ x: 0, y: 0 }) // Reset position
    }
  }, [isOpen, imageUrl, storeData])

  // Theo dõi thay đổi scale để reset position khi scale = 1
  useEffect(() => {
    if (scale === 1) {
      setPosition({ x: 0, y: 0 })
    }
  }, [scale])

  // Thêm event listeners toàn cục cho mouse move và mouse up
  useEffect(() => {
    const handleGlobalMouseMove = (e: MouseEvent) => {
      if (isDragging && scale > 1) {
        e.preventDefault()
        const newX = e.clientX - dragStartRef.current.x
        const newY = e.clientY - dragStartRef.current.y
        setPosition({ x: newX, y: newY })
      }
    }

    const handleGlobalMouseUp = () => {
      if (isDragging) {
        setIsDragging(false)
      }
    }

    if (isDragging) {
      window.addEventListener("mousemove", handleGlobalMouseMove)
      window.addEventListener("mouseup", handleGlobalMouseUp)
    }

    return () => {
      window.removeEventListener("mousemove", handleGlobalMouseMove)
      window.removeEventListener("mouseup", handleGlobalMouseUp)
    }
  }, [isDragging, scale])

  if (!isOpen) return null

  const handleZoomIn = () => {
    setScale((prev) => Math.min(prev + 0.25, 3))
  }

  const handleZoomOut = () => {
    const newScale = Math.max(scale - 0.25, 1)
    setScale(newScale)
    if (newScale === 1) {
      setPosition({ x: 0, y: 0 }) // Reset position when zoomed out to original size
    }
  }

  const handleBackdropClick = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget) {
      onClose()
    }
  }

  const toggleInfo = () => {
    setShowInfo((prev) => !prev)
  }

  // Xử lý sự kiện chuột để kéo ảnh
  const handleMouseDown = (e: React.MouseEvent) => {
    if (scale > 1) {
      e.preventDefault()
      setIsDragging(true)
      dragStartRef.current = { x: e.clientX - position.x, y: e.clientY - position.y }
    }
  }

  // Thêm tính năng double click để zoom
  const handleDoubleClick = (e: React.MouseEvent) => {
    if (scale > 1) {
      // Nếu đã zoom, double click sẽ reset về kích thước gốc
      setScale(1)
      setPosition({ x: 0, y: 0 })
    } else {
      // Nếu chưa zoom, double click sẽ zoom đến mức 2x
      setScale(2)

      // Tính toán vị trí zoom dựa trên vị trí click
      if (imageRef.current && containerRef.current) {
        const rect = imageRef.current.getBoundingClientRect()
        const containerRect = containerRef.current.getBoundingClientRect()

        // Tính toán tỷ lệ vị trí click so với kích thước ảnh
        const relativeX = (e.clientX - rect.left) / rect.width
        const relativeY = (e.clientY - rect.top) / rect.height

        // Tính toán vị trí mới sau khi zoom
        const scaledWidth = rect.width * 2
        const scaledHeight = rect.height * 2

        // Tính toán vị trí để giữ điểm click ở cùng vị trí tương đối
        const newX = containerRect.width / 2 - scaledWidth * relativeX
        const newY = containerRect.height / 2 - scaledHeight * relativeY

        setPosition({ x: newX, y: newY })
      }
    }
  }

  return (
    <div className="fixed inset-0 bg-black/70 z-50 flex items-center justify-center" onClick={handleBackdropClick}>
      <div className="relative max-w-[95vw] max-h-[95vh] flex flex-col md:flex-row gap-4 items-center">
        {/* Các nút điều khiển được đặt ở phía trên hình ảnh */}
        <div className="absolute top-2 left-2 flex gap-2 z-10">
          <Button size="icon" variant="secondary" onClick={handleZoomIn} className="bg-white/80 hover:bg-white">
            <ZoomIn className="h-4 w-4" />
            <span className="sr-only">Phóng to</span>
          </Button>
          <Button size="icon" variant="secondary" onClick={handleZoomOut} className="bg-white/80 hover:bg-white">
            <ZoomOut className="h-4 w-4" />
            <span className="sr-only">Thu nhỏ</span>
          </Button>
          {storeData && (
            <Button
              size="icon"
              variant="secondary"
              onClick={toggleInfo}
              className={`bg-white/80 hover:bg-white ${showInfo ? "text-primary" : ""}`}
            >
              <Building2 className="h-4 w-4" />
              <span className="sr-only">Thông tin</span>
            </Button>
          )}
        </div>

        {/* Nút đóng được đặt ở góc trên bên phải */}
        <div className="absolute top-2 right-2 z-10">
          <Button
            size="icon"
            variant="destructive"
            onClick={onClose}
            className="bg-white/80 hover:bg-white text-red-500"
          >
            <X className="h-4 w-4" />
            <span className="sr-only">Đóng</span>
          </Button>
        </div>

        <div
          ref={containerRef}
          className="bg-white/10 backdrop-blur-sm p-1 rounded-lg overflow-hidden max-w-full max-h-[85vh]"
        >
          <img
            ref={imageRef}
            src={imageUrl || "/placeholder.svg"}
            alt={imageTitle || "Hình ảnh xem trước"}
            className="max-w-full max-h-[85vh] object-contain select-none"
            style={{
              transform: `translate(${position.x}px, ${position.y}px) scale(${scale})`,
              cursor: scale > 1 ? (isDragging ? "grabbing" : "grab") : "default",
              transition: isDragging ? "none" : "transform 0.2s",
            }}
            onMouseDown={handleMouseDown}
            onDoubleClick={handleDoubleClick}
            draggable="false"
          />
          {imageTitle && !storeData && (
            <div className="bg-black/50 text-white text-sm py-2 px-4 text-center absolute bottom-0 left-0 right-0">
              {imageTitle}
            </div>
          )}
        </div>

        {storeData && showInfo && (
          <Card className="bg-white/90 backdrop-blur-sm shadow-lg max-w-md w-full md:w-80 self-start mt-12 md:mt-0">
            <CardContent className="p-4 space-y-4">
              <div>
                <h3 className="text-lg font-semibold mb-2">Thông tin trích xuất</h3>
                <p className="text-xs text-muted-foreground mb-4">Từ hình ảnh: {imageTitle}</p>
              </div>

              <div className="space-y-3">
                <div>
                  <div className="flex items-center mb-1">
                    <Building2 className="h-4 w-4 mr-1 text-primary" />
                    <p className="text-sm font-medium">Tên cửa hàng:</p>
                  </div>
                  <p className="text-base bg-white/50 p-2 rounded">{storeData.name || "Không xác định"}</p>
                </div>

                <div>
                  <div className="flex items-center mb-1">
                    <MapPin className="h-4 w-4 mr-1 text-muted-foreground" />
                    <p className="text-sm font-medium">Địa chỉ:</p>
                  </div>
                  <p className="text-base bg-white/50 p-2 rounded">{storeData.address || "Không xác định"}</p>
                </div>

                <div>
                  <div className="flex items-center mb-1">
                    <Phone className="h-4 w-4 mr-1 text-muted-foreground" />
                    <p className="text-sm font-medium">Số điện thoại:</p>
                  </div>
                  <p className="text-base bg-white/50 p-2 rounded">{storeData.tel || "Không xác định"}</p>
                </div>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}


"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { HistoryItem } from "@/hooks/actions"
import { formatDistanceToNow } from "date-fns"
import { vi } from "date-fns/locale"
import { Eye, Plus } from "lucide-react"

interface HistorySidebarProps {
  sampleImages: HistoryItem[]
  onSelect: (url: string, name: string) => void
  onPreview: (url: string, name: string) => void
  isLoading: boolean
}

export function HistorySidebar({ sampleImages, onSelect, onPreview, isLoading }: HistorySidebarProps) {
  return (
    <Card className="h-full sticky top-4">
      <CardHeader className="pb-3">
        <CardTitle className="text-lg">Hình ảnh mẫu</CardTitle>
        <p className="text-xs text-muted-foreground">Nhấp vào hình ảnh để thêm vào danh sách</p>
      </CardHeader>
      <CardContent className="space-y-4 overflow-y-auto" style={{ maxHeight: "calc(100vh - 180px)" }}>
        {sampleImages.length === 0 ? (
          <div className="text-center py-8 text-muted-foreground">
            <p>Chưa có hình ảnh mẫu nào</p>
          </div>
        ) : (
          <div className="grid grid-cols-2 gap-3">
            {sampleImages.map((item) => (
              <div key={item.id} className="cursor-pointer group relative">
                <div className="aspect-square rounded-md overflow-hidden border bg-muted">
                  <img
                    src={item.url || "/placeholder.svg"}
                    alt={item.name}
                    className="h-full w-full object-cover transition-transform group-hover:scale-105"
                  />
                </div>
                <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center gap-2">
                  <button
                    onClick={() => !isLoading && onSelect(item.url, item.name)}
                    className="text-white text-xs font-medium px-2 py-1 bg-black/50 rounded flex items-center gap-1 hover:bg-black/70"
                    disabled={isLoading}
                  >
                    <Plus className="h-3 w-3" />
                    Thêm
                  </button>
                  <button
                    onClick={(e) => {
                      e.stopPropagation()
                      onPreview(item.url, item.name)
                    }}
                    className="text-white text-xs font-medium px-2 py-1 bg-black/50 rounded flex items-center gap-1 hover:bg-black/70"
                  >
                    <Eye className="h-3 w-3" />
                    Xem
                  </button>
                </div>
                {/* <p className="text-xs truncate mt-1">{item.name}</p> */}
                {/* <p className="text-xs text-muted-foreground">
                  {formatDistanceToNow(item.timestamp, { addSuffix: true, locale: vi })}
                </p> */}
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}

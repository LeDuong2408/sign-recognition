"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
import { formatDistanceToNow } from "date-fns"
import { vi } from "date-fns/locale"
import { RefreshCw } from "lucide-react"
import { HistoryItem } from "@/hooks/actions"

interface UploadHistoryProps {
  history: HistoryItem[]
  onSelect: (urls: string[]) => void
  onRefresh: () => void
}

export function UploadHistory({ history, onSelect, onRefresh }: UploadHistoryProps) {
  const [selectedItems, setSelectedItems] = useState<string[]>([])
  const [isRefreshing, setIsRefreshing] = useState(false)

  const toggleSelect = (id: string) => {
    setSelectedItems((prev) => {
      if (prev.includes(id)) {
        return prev.filter((item) => item !== id)
      } else {
        return [...prev, id]
      }
    })
  }

  const handleSelectAll = () => {
    if (selectedItems.length === history.length) {
      setSelectedItems([])
    } else {
      setSelectedItems(history.map((item) => item.id))
    }
  }

  const handleUseSelected = () => {
    const selectedUrls = history.filter((item) => selectedItems.includes(item.id)).map((item) => item.url)
    onSelect(selectedUrls)
  }

  const handleRefresh = async () => {
    setIsRefreshing(true)
    await onRefresh()
    setIsRefreshing(false)
  }

  if (history.length === 0) {
    return (
      <div className="text-center py-8 text-muted-foreground flex flex-col items-center gap-4">
        <p>Chưa có hình ảnh nào được tải lên</p>
        <Button variant="outline" size="sm" onClick={handleRefresh} disabled={isRefreshing}>
          {isRefreshing ? (
            <>
              <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
              Đang làm mới...
            </>
          ) : (
            <>
              <RefreshCw className="mr-2 h-4 w-4" />
              Làm mới
            </>
          )}
        </Button>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-medium">Lịch sử hình ảnh đã tải lên</h3>
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" onClick={handleRefresh} disabled={isRefreshing}>
            {isRefreshing ? (
              <>
                <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
                Đang làm mới...
              </>
            ) : (
              <>
                <RefreshCw className="mr-2 h-4 w-4" />
                Làm mới
              </>
            )}
          </Button>
          <Button variant="outline" size="sm" onClick={handleSelectAll}>
            {selectedItems.length === history.length ? "Bỏ chọn tất cả" : "Chọn tất cả"}
          </Button>
          <Button size="sm" onClick={handleUseSelected} disabled={selectedItems.length === 0}>
            Sử dụng đã chọn
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
        {history.map((item) => (
          <div key={item.id} className="relative group">
            <div className="aspect-square rounded-md overflow-hidden border bg-muted">
              <img src={item.url || "/placeholder.svg"} alt={item.name} className="h-full w-full object-cover" />
            </div>
            <div className="absolute top-2 left-2">
              <Checkbox
                checked={selectedItems.includes(item.id)}
                onCheckedChange={() => toggleSelect(item.id)}
                className="bg-white/80 border-gray-400"
              />
            </div>
            <div className="mt-1 flex flex-col">
              <p className="text-xs truncate">{item.name}</p>
              <p className="text-xs text-muted-foreground">
                {formatDistanceToNow(item.timestamp, { addSuffix: true, locale: vi })}
              </p>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

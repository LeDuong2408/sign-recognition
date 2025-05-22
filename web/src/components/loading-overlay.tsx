"use client"

import { Loader2 } from "lucide-react"

interface LoadingOverlayProps {
  isLoading: boolean
  message?: string
}

export function LoadingOverlay({ isLoading, message = "Đang xử lý..." }: LoadingOverlayProps) {
  if (!isLoading) return null

  return (
    <div className="fixed inset-0 bg-black/30 backdrop-blur-[2px] z-50 flex items-center justify-center">
      <div className="bg-white/90 rounded-lg p-4 shadow-lg flex items-center gap-3 max-w-md mx-4">
        <Loader2 className="h-6 w-6 text-primary animate-spin" />
        <p className="text-sm font-medium">{message}</p>
      </div>
    </div>
  )
}

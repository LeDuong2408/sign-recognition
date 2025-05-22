"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type { StoreData } from "@/types/store-data"
import { Building2, MapPin, Phone, ImageIcon } from "lucide-react"
import { Button } from "@/components/ui/button"

interface ResultsDisplayProps {
  results: StoreData[]
  images?: { url: string; name: string }[]
  onPreviewImage?: (url: string, title: string, storeData: StoreData) => void
}

export function ResultsDisplay({ results, images = [], onPreviewImage }: ResultsDisplayProps) {
  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">Kết quả trích xuất</h2>

      <div className="grid gap-4 md:grid-cols-2">
        {results.map((store, index) => {
          const image = images[index]

          return (
            <Card key={index}>
              <CardHeader className="pb-2">
                <CardTitle className="text-lg flex items-center justify-between">
                  <div className="flex items-center">
                    <Building2 className="h-5 w-5 mr-2 text-primary" />
                    {store.name || "Không xác định"}
                  </div>
                  {image && onPreviewImage && (
                    <Button
                      variant="outline"
                      size="sm"
                      className="h-8 flex items-center gap-1"
                      onClick={() => onPreviewImage(image.url, image.name, store)}
                    >
                      <ImageIcon className="h-4 w-4" />
                      Xem ảnh
                    </Button>
                  )}
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                <div className="flex items-start">
                  <MapPin className="h-4 w-4 mr-2 text-muted-foreground mt-0.5" />
                  <p className="text-sm">{store.address || "Không xác định"}</p>
                </div>
                <div className="flex items-center">
                  <Phone className="h-4 w-4 mr-2 text-muted-foreground" />
                  <p className="text-sm">{store.tel || "Không xác định"}</p>
                </div>
              </CardContent>
            </Card>
          )
        })}
      </div>
    </div>
  )
}

"use server"

import type { StoreData } from "@/types/store-data"

export interface HistoryItem {
  id: string
  url: string
  name: string
  timestamp: number
}

// Hàm tải lên hình ảnh và trích xuất dữ liệu
export async function uploadAndExtract(formData: FormData): Promise<{ results: StoreData[] }> {
  const files: File[] = []
  const entries = Array.from(formData.entries())

  for (const [key, value] of entries) {
    if (key.startsWith("image-") && value instanceof File) {
      files.push(value)
    }
  }

  if (files.length === 0) {
    throw new Error("Không có file nào được chọn")
  }

  try {
    // 1. Tải lên hình ảnh lên Vercel Blob
    // const uploadPromises = files.map(async (file) => {
    //   // Tạo tên file duy nhất để tránh trùng lặp
    //   const uniqueFileName = `sign-board-ocr/${Date.now()}-${Math.random().toString(36).substring(2, 15)}-${file.name}`

    //   // Tải lên Vercel Blob
    //   const blob = await put(uniqueFileName, file, {
    //     access: "public",
    //     addRandomSuffix: false, // Đã thêm suffix ngẫu nhiên ở trên
    //   })

    //   return blob.url
    // })
    const CLOUD_NAME='dnclfveb3'
    const UPLOAD_PRESET = 'sign-board '
    const uploadPromises = files.map(async (file) => {
      const timestamp = Date.now()
      const randomStr = Math.random().toString(36).substring(2, 10)
      const uploadForm = new FormData()
      uploadForm.append("file", file)
      uploadForm.append("upload_preset", UPLOAD_PRESET)
      uploadForm.append("folder", "sign-board-ocr") // optional, đã có trong tên
      uploadForm.append("public_id", `${timestamp}-${randomStr}-${file.name}`)

      const response = await fetch(`https://api.cloudinary.com/v1_1/${CLOUD_NAME}/image/upload`, {
        method: "POST",
        body: uploadForm
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(`Upload thất bại: ${data.error?.message || response.statusText}`)
      }

      return data.secure_url as string // Link ảnh HTTPS trên Cloudinary
    })

    const imageUrls = await Promise.all(uploadPromises)
    // Mô phỏng thời gian xử lý OCR
    await new Promise((resolve) => setTimeout(resolve, 2000))

    // 2. Trích xuất dữ liệu từ các URL hình ảnh
    const results = await extractSignBoardData(imageUrls)

    return {
      results,
    }
  } catch (error) {
    console.error("Lỗi khi xử lý hình ảnh:", error)
    throw new Error("Không thể xử lý hình ảnh. Vui lòng thử lại sau.")
  }
}

function encodeBasicAuth(key: string, secret: string): string {
  return Buffer.from(`${key}:${secret}`).toString("base64")
}


// Hàm lấy lịch sử ảnh từ Cloudinary
export async function getUploadHistory(): Promise<HistoryItem[]> {
    // Trả về dữ liệu mẫu khi lỗi
    return [
      {
        id: "sample-1",
        url: "/example-1.png?height=400&width=600",
        name: "Mẫu 1",
        timestamp: Date.now() - 86400000,
      },
      {
        id: "sample-2",
        url: "/example-2.png?height=400&width=600",
        name: "Mẫu 2",
        timestamp: Date.now() - 172800000,
      },
      {
        id: "sample-3",
        url: "/example-3.png?height=400&width=600",
        name: "Mẫu 3",
        timestamp: Date.now() - 259200000,
      },
      {
        id: "sample-4",
        url: "/example-4.png?height=400&width=600",
        name: "Mẫu 4",
        timestamp: Date.now() - 259200000,
      },
    ]
}


// // Hàm lấy lịch sử ảnh từ Cloudinary
// export async function getUploadHistory(): Promise<HistoryItem[]> {
//   const API_KEY = '816545419225756'
//   const API_SECRET = 'H77K8Hig3fSOLth6sYQc6QlUw54'
//   const CLOUD_NAME='dnclfveb3'

//   try {
//     const authHeader = encodeBasicAuth(API_KEY, API_SECRET)

//     const folder = "sign-board-ocr" // Tên folder Cloudinary nếu bạn đã dùng

//     const res = await fetch(
//       `https://api.cloudinary.com/v1_1/${CLOUD_NAME}/resources/image/upload?prefix=${folder}/`,
//       {
//         headers: {
//           Authorization: `Basic ${authHeader}`,
//         },
//       }
//     )

//     if (!res.ok) {
//       throw new Error("Không thể lấy danh sách ảnh từ Cloudinary")
//     }

//     const data = await res.json()

//     const historyItems: HistoryItem[] = data.resources.map((item: any) => {
//       const fileName = item.public_id.split("/").pop() || "unknown"
//       const originalName = fileName.split("-").slice(2).join("-") || fileName

//       return {
//         id: item.asset_id,
//         url: item.secure_url,
//         name: originalName,
//         timestamp: new Date(item.created_at).getTime(),
//       }
//     })

//     return historyItems.sort((a, b) => b.timestamp - a.timestamp)
//   } catch (error) {
//     console.error("Lỗi khi lấy lịch sử Cloudinary:", error)

//     // Trả về dữ liệu mẫu khi lỗi
//     return [
//       {
//         id: "sample-1",
//         url: "/placeholder.svg?height=400&width=600",
//         name: "Cửa hàng mẫu 1",
//         timestamp: Date.now() - 86400000,
//       },
//       {
//         id: "sample-2",
//         url: "/placeholder.svg?height=400&width=600",
//         name: "Cửa hàng mẫu 2",
//         timestamp: Date.now() - 172800000,
//       },
//       {
//         id: "sample-3",
//         url: "/placeholder.svg?height=400&width=600",
//         name: "Cửa hàng mẫu 3",
//         timestamp: Date.now() - 259200000,
//       },
//     ]
//   }
// }

export async function extractSignBoardData(imageUrls: string[]): Promise<StoreData[]> {
  try {
    // Gọi API OCR thực tế
    const OCR_API_URL = process.env.OCR_API_URL || ""
    // const OCR_API_KEY = process.env.OCR_API_KEY

    // if (!OCR_API_KEY) {
    //   throw new Error("OCR API key không được cấu hình")
    // }

    // Gọi API OCR với các URL hình ảnh

    const response = await fetch(OCR_API_URL, { 
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        // Authorization: `Bearer ${OCR_API_KEY}`,
      },
      body: JSON.stringify({
        img_urls: imageUrls,
        options: {
          language: "vi",
          detectStoreInfo: true,
        },
      }),
    })

    if (!response.ok) {
      const errorText = await response.text()
      console.error("Lỗi từ API OCR:", errorText)
      throw new Error(`Lỗi khi gọi API OCR: ${response.status}`)
    }

    const rsp = await response.json()
    // Chuyển đổi kết quả từ API sang định dạng StoreData
    // Điều chỉnh theo cấu trúc phản hồi thực tế từ API của bạn
    const results: StoreData[] = rsp.data
    // .map((result: any) => ({
    //   name: result.name || "",
    //   address: result.address || "",
    //   tel: result.tel || "",
    // }))

    return results
  } catch (error) {
    console.error("Lỗi khi trích xuất dữ liệu:", error)

    // Nếu có lỗi, trả về dữ liệu mẫu để kiểm tra giao diện
    // Trong môi trường sản xuất, bạn có thể muốn ném lỗi thay vì trả về dữ liệu mẫu
    return imageUrls.map((_, i) => ({
      name: `Cửa hàng mẫu ${i + 1}`,
      address: `Địa chỉ mẫu ${i + 1}`,
      tel: `0${Math.floor(Math.random() * 900000000) + 100000000}`,
    }))
  }
}


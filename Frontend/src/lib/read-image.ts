/** Baca dimensi asli sebuah gambar dari object URL-nya. */
export function readImageSize(url: string): Promise<{ width: number; height: number }> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve({ width: image.naturalWidth, height: image.naturalHeight });
    image.onerror = () => reject(new Error("gambar tidak dapat dibaca"));
    image.src = url;
  });
}

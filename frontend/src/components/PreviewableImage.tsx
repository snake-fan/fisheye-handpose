import { Maximize2, X } from "lucide-react";
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  type ImgHTMLAttributes,
  type ReactNode,
} from "react";
import { createPortal } from "react-dom";

interface PreviewableImageProps extends ImgHTMLAttributes<HTMLImageElement> {
  previewOverlay?: ReactNode;
}

interface PreviewState {
  alt: string;
  imageProps: ImgHTMLAttributes<HTMLImageElement>;
  overlay?: ReactNode;
}

interface PreviewController {
  openPreview: (preview: PreviewState, trigger: HTMLButtonElement) => void;
}

const PreviewContext = createContext<PreviewController | null>(null);

export function ImagePreviewProvider({ children }: { children: ReactNode }) {
  const [preview, setPreview] = useState<PreviewState | null>(null);
  const titleId = useId();
  const activeTriggerRef = useRef<HTMLButtonElement | null>(null);
  const closeButtonRef = useRef<HTMLButtonElement>(null);
  const dialogRef = useRef<HTMLDivElement>(null);
  const open = preview !== null;

  const closePreview = useCallback(() => setPreview(null), []);
  const openPreview = useCallback((nextPreview: PreviewState, trigger: HTMLButtonElement) => {
    activeTriggerRef.current = trigger;
    setPreview(nextPreview);
  }, []);
  const controller = useMemo(() => ({ openPreview }), [openPreview]);

  useEffect(() => {
    if (!open) return;

    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    closeButtonRef.current?.focus();

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        closePreview();
        return;
      }

      if (event.key !== "Tab") return;
      const focusable = dialogRef.current?.querySelectorAll<HTMLElement>(
        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])',
      );
      if (!focusable?.length) return;

      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("keydown", handleKeyDown);
      document.body.style.overflow = previousOverflow;
      if (activeTriggerRef.current?.isConnected) activeTriggerRef.current.focus();
    };
  }, [closePreview, open]);

  const previewTitle = preview?.alt || "图片预览";

  return (
    <PreviewContext.Provider value={controller}>
      <div
        className="image-preview-root"
        data-image-preview-root=""
        inert={open}
        aria-hidden={open || undefined}
      >
        {children}
      </div>

      {preview && createPortal(
        <div
          ref={dialogRef}
          className="image-preview-backdrop"
          role="dialog"
          aria-modal="true"
          aria-labelledby={titleId}
          onMouseDown={(event) => {
            if (event.target === event.currentTarget) closePreview();
          }}
        >
          <section className="image-preview-dialog">
            <header>
              <span id={titleId}>{previewTitle}</span>
              <button
                ref={closeButtonRef}
                type="button"
                aria-label="关闭图片预览"
                title="关闭 (Esc)"
                onClick={closePreview}
              >
                <X aria-hidden="true" />
              </button>
            </header>
            <div className={`image-preview-canvas ${preview.overlay ? "with-overlay" : ""}`}>
              <button
                type="button"
                className="image-preview-media"
                aria-label={`返回：${previewTitle}`}
                title="再次点击返回"
                onClick={closePreview}
              >
                <img
                  src={preview.imageProps.src}
                  srcSet={preview.imageProps.srcSet}
                  sizes={preview.imageProps.sizes}
                  alt={`${previewTitle}（大图预览）`}
                />
                {preview.overlay && <span className="image-preview-overlay">{preview.overlay}</span>}
              </button>
            </div>
          </section>
        </div>,
        document.body,
      )}
    </PreviewContext.Provider>
  );
}

export function PreviewableImage({
  alt = "",
  previewOverlay,
  ...imageProps
}: PreviewableImageProps) {
  const controller = useContext(PreviewContext);
  const triggerRef = useRef<HTMLButtonElement>(null);

  if (!controller) {
    return (
      <ImagePreviewProvider>
        <PreviewableImage {...imageProps} alt={alt} previewOverlay={previewOverlay} />
      </ImagePreviewProvider>
    );
  }

  const previewTitle = alt || "图片预览";
  return (
    <button
      ref={triggerRef}
      type="button"
      className="image-preview-trigger"
      aria-haspopup="dialog"
      aria-label={`放大预览：${previewTitle}`}
      title="点击放大预览"
      onClick={() => {
        if (!triggerRef.current) return;
        controller.openPreview({ alt: previewTitle, imageProps, overlay: previewOverlay }, triggerRef.current);
      }}
    >
      <img {...imageProps} alt={alt} />
      <span className="image-preview-hint" aria-hidden="true">
        <Maximize2 />
      </span>
    </button>
  );
}

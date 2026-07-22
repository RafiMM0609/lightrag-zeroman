import { useAuthStore } from '@/stores/state'
import { ZapIcon } from 'lucide-react'

interface BrandLogoProps {
  className?: string
  iconSizeClassName?: string
}

export default function BrandLogo({ className = 'h-6', iconSizeClassName }: BrandLogoProps) {
  const { webuiLogoType, webuiLogoUrl, webuiTitle } = useAuthStore()

  const logoType = webuiLogoType || 'default'
  const appName = webuiTitle || 'LightRAG'

  if (logoType === 'zeroman') {
    // Beautiful serif font design with a solid rectangle block
    return (
      <div className={`flex items-center gap-2 select-none ${className}`}>
        <span 
          className="text-2xl font-normal tracking-wide text-emerald-400 leading-none" 
          style={{ fontFamily: "'Playfair Display', 'Lora', 'Georgia', serif" }}
        >
          zeroman
        </span>
        <div 
          className="w-[10px] h-[22px] bg-emerald-400 rounded-sm self-center ml-0.5" 
          aria-hidden="true" 
        />
      </div>
    )
  }

  if (logoType === 'custom' && webuiLogoUrl) {
    return (
      <img 
        src={webuiLogoUrl} 
        alt={appName} 
        className={className} 
      />
    )
  }

  if (logoType === 'text') {
    return (
      <span className="font-bold text-lg leading-none">
        {appName}
      </span>
    )
  }

  // Default: SVG file with ZapIcon fallback or both
  return (
    <div className="flex items-center gap-2">
      <img 
        src="logo.svg" 
        alt={`${appName} Logo`} 
        className={className} 
        onError={(e) => {
          // Fallback if logo.svg is not found/accessible
          e.currentTarget.style.display = 'none'
        }}
      />
      <ZapIcon className={iconSizeClassName || "size-5 text-emerald-400"} aria-hidden="true" />
    </div>
  )
}

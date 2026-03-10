import { Box, Skeleton } from '@mui/material';

export function ShimmerLoader() {
  return (
    <Box
      sx={{
        minHeight: '100vh',
        background: 'linear-gradient(135deg, #F8FAFC 0%, #EFF6FF 50%, #ECFDF5 100%)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        p: 4,
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      {/* Animated background orbs */}
      <Box
        sx={{
          position: 'absolute',
          top: '20%',
          left: '10%',
          width: '288px',
          height: '288px',
          bgcolor: 'rgba(59, 130, 246, 0.1)',
          borderRadius: '50%',
          filter: 'blur(60px)',
          animation: 'pulse 3s ease-in-out infinite',
        }}
      />
      <Box
        sx={{
          position: 'absolute',
          bottom: '20%',
          right: '10%',
          width: '384px',
          height: '384px',
          bgcolor: 'rgba(16, 185, 129, 0.1)',
          borderRadius: '50%',
          filter: 'blur(60px)',
          animation: 'pulse 3s ease-in-out infinite',
          animationDelay: '1s',
        }}
      />

      <Box
        sx={{
          textAlign: 'center',
          position: 'relative',
          zIndex: 10,
          maxWidth: '768px',
          width: '100%',
        }}
      >
        {/* Badge Skeleton */}
        <Box sx={{ display: 'flex', justifyContent: 'center', mb: 3 }}>
          <Skeleton
            variant="rounded"
            width={280}
            height={40}
            sx={{
              bgcolor: 'rgba(255, 255, 255, 0.8)',
              borderRadius: '20px',
              animation: 'shimmer 2s ease-in-out infinite',
              background: 'linear-gradient(90deg, rgba(241, 245, 249, 0.8) 25%, rgba(226, 232, 240, 0.8) 50%, rgba(241, 245, 249, 0.8) 75%)',
              backgroundSize: '200% 100%',
              '@keyframes shimmer': {
                '0%': { backgroundPosition: '200% 0' },
                '100%': { backgroundPosition: '-200% 0' },
              },
            }}
          />
        </Box>

        {/* Main heading Skeleton */}
        <Box sx={{ mb: 2 }}>
          <Skeleton
            variant="rounded"
            width="90%"
            height={60}
            sx={{
              mx: 'auto',
              mb: 2,
              bgcolor: 'rgba(255, 255, 255, 0.9)',
              borderRadius: '12px',
              animation: 'shimmer 2s ease-in-out infinite',
              background: 'linear-gradient(90deg, rgba(241, 245, 249, 0.9) 25%, rgba(226, 232, 240, 0.9) 50%, rgba(241, 245, 249, 0.9) 75%)',
              backgroundSize: '200% 100%',
              animationDelay: '0.2s',
            }}
          />
          <Skeleton
            variant="rounded"
            width="70%"
            height={60}
            sx={{
              mx: 'auto',
              bgcolor: 'rgba(59, 130, 246, 0.15)',
              borderRadius: '12px',
              animation: 'shimmer 2s ease-in-out infinite',
              background: 'linear-gradient(90deg, rgba(191, 219, 254, 0.4) 25%, rgba(147, 197, 253, 0.5) 50%, rgba(191, 219, 254, 0.4) 75%)',
              backgroundSize: '200% 100%',
              animationDelay: '0.3s',
            }}
          />
        </Box>

        {/* Description Skeleton */}
        <Box sx={{ mb: 5, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 1.5 }}>
          <Skeleton
            variant="rounded"
            width="80%"
            height={24}
            sx={{
              bgcolor: 'rgba(255, 255, 255, 0.7)',
              borderRadius: '8px',
              animation: 'shimmer 2s ease-in-out infinite',
              background: 'linear-gradient(90deg, rgba(241, 245, 249, 0.7) 25%, rgba(226, 232, 240, 0.7) 50%, rgba(241, 245, 249, 0.7) 75%)',
              backgroundSize: '200% 100%',
              animationDelay: '0.4s',
            }}
          />
          <Skeleton
            variant="rounded"
            width="75%"
            height={24}
            sx={{
              bgcolor: 'rgba(255, 255, 255, 0.7)',
              borderRadius: '8px',
              animation: 'shimmer 2s ease-in-out infinite',
              background: 'linear-gradient(90deg, rgba(241, 245, 249, 0.7) 25%, rgba(226, 232, 240, 0.7) 50%, rgba(241, 245, 249, 0.7) 75%)',
              backgroundSize: '200% 100%',
              animationDelay: '0.5s',
            }}
          />
        </Box>

        {/* Button Skeleton */}
        <Box sx={{ display: 'flex', justifyContent: 'center', mb: 6 }}>
          <Skeleton
            variant="rounded"
            width={340}
            height={64}
            sx={{
              bgcolor: 'rgba(59, 130, 246, 0.3)',
              borderRadius: '16px',
              boxShadow: '0 20px 40px rgba(59, 130, 246, 0.2)',
              animation: 'shimmer 2s ease-in-out infinite, pulse 2s ease-in-out infinite',
              background: 'linear-gradient(90deg, rgba(96, 165, 250, 0.4) 25%, rgba(59, 130, 246, 0.5) 50%, rgba(96, 165, 250, 0.4) 75%)',
              backgroundSize: '200% 100%',
              animationDelay: '0.6s',
            }}
          />
        </Box>

        {/* Feature highlights Skeleton */}
        <Box
          sx={{
            display: 'grid',
            gridTemplateColumns: { xs: '1fr', md: 'repeat(3, 1fr)' },
            gap: 2,
            maxWidth: '672px',
            mx: 'auto',
          }}
        >
          {[0, 1, 2].map((idx) => (
            <Box
              key={idx}
              sx={{
                bgcolor: 'rgba(255, 255, 255, 0.6)',
                backdropFilter: 'blur(8px)',
                borderRadius: '12px',
                p: 3,
                border: '1px solid rgba(226, 232, 240, 0.8)',
              }}
            >
              <Skeleton
                variant="circular"
                width={48}
                height={48}
                sx={{
                  mx: 'auto',
                  mb: 2,
                  bgcolor: 'rgba(59, 130, 246, 0.1)',
                  animation: 'shimmer 2s ease-in-out infinite',
                  background: 'linear-gradient(90deg, rgba(191, 219, 254, 0.3) 25%, rgba(147, 197, 253, 0.4) 50%, rgba(191, 219, 254, 0.3) 75%)',
                  backgroundSize: '200% 100%',
                  animationDelay: `${0.7 + idx * 0.1}s`,
                }}
              />
              <Skeleton
                variant="rounded"
                width="80%"
                height={20}
                sx={{
                  mx: 'auto',
                  mb: 1.5,
                  bgcolor: 'rgba(15, 23, 42, 0.1)',
                  borderRadius: '6px',
                  animation: 'shimmer 2s ease-in-out infinite',
                  background: 'linear-gradient(90deg, rgba(226, 232, 240, 0.5) 25%, rgba(203, 213, 225, 0.6) 50%, rgba(226, 232, 240, 0.5) 75%)',
                  backgroundSize: '200% 100%',
                  animationDelay: `${0.8 + idx * 0.1}s`,
                }}
              />
              <Skeleton
                variant="rounded"
                width="60%"
                height={16}
                sx={{
                  mx: 'auto',
                  bgcolor: 'rgba(100, 116, 139, 0.1)',
                  borderRadius: '6px',
                  animation: 'shimmer 2s ease-in-out infinite',
                  background: 'linear-gradient(90deg, rgba(226, 232, 240, 0.4) 25%, rgba(203, 213, 225, 0.5) 50%, rgba(226, 232, 240, 0.4) 75%)',
                  backgroundSize: '200% 100%',
                  animationDelay: `${0.9 + idx * 0.1}s`,
                }}
              />
            </Box>
          ))}
        </Box>

        {/* Loading text */}
        <Box
          sx={{
            mt: 8,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 2,
          }}
        >
          <Box
            sx={{
              width: 8,
              height: 8,
              bgcolor: '#3B82F6',
              borderRadius: '50%',
              animation: 'pulse 1.5s ease-in-out infinite',
            }}
          />
          <Box
            sx={{
              fontSize: '14px',
              fontWeight: 600,
              color: '#64748B',
              letterSpacing: '0.05em',
              animation: 'pulse 2s ease-in-out infinite',
            }}
          >
            Loading Environmental Dashboard...
          </Box>
        </Box>
      </Box>

      <style>
        {`
          @keyframes shimmer {
            0% { background-position: 200% 0; }
            100% { background-position: -200% 0; }
          }
          @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
          }
        `}
      </style>
    </Box>
  );
}

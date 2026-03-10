import { Box, Skeleton, LinearProgress, Stepper, Step, StepLabel } from '@mui/material';
import { Check } from 'lucide-react';

interface ShimmerLoaderProps {
  progress: number;
}

const loadingSteps = [
  { label: 'Initializing Dashboard', threshold: 20 },
  { label: 'Loading Environmental Data', threshold: 40 },
  { label: 'Rendering Radar Chart', threshold: 60 },
  { label: 'Configuring Metrics', threshold: 80 },
  { label: 'Finalizing Components', threshold: 100 },
];

export function ShimmerLoader({ progress }: ShimmerLoaderProps) {
  const activeStep = loadingSteps.findIndex(step => progress < step.threshold);
  const currentStep = activeStep === -1 ? loadingSteps.length : activeStep;

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
        {/* Progress Steps */}
        <Box
          sx={{
            mb: 6,
            bgcolor: 'rgba(255, 255, 255, 0.8)',
            backdropFilter: 'blur(12px)',
            borderRadius: '20px',
            p: 4,
            border: '1px solid rgba(226, 232, 240, 0.8)',
            boxShadow: '0 10px 40px rgba(0, 0, 0, 0.08)',
          }}
        >
          <Box sx={{ mb: 4, textAlign: 'center' }}>
            <Box
              sx={{
                display: 'inline-flex',
                alignItems: 'center',
                justifyContent: 'center',
                width: 64,
                height: 64,
                background: 'linear-gradient(135deg, #3B82F6 0%, #2563EB 100%)',
                borderRadius: '16px',
                mb: 2,
                boxShadow: '0 8px 24px rgba(59, 130, 246, 0.3)',
                animation: 'pulse 2s ease-in-out infinite',
              }}
            >
              <Box
                sx={{
                  fontSize: '32px',
                  fontWeight: 700,
                  color: 'white',
                }}
              >
                {Math.round(progress)}%
              </Box>
            </Box>
          </Box>

          {/* Custom Stepper */}
          <Box sx={{ width: '100%' }}>
            {loadingSteps.map((step, index) => (
              <Box
                key={index}
                sx={{
                  display: 'flex',
                  alignItems: 'center',
                  mb: index < loadingSteps.length - 1 ? 2.5 : 0,
                  position: 'relative',
                }}
              >
                {/* Step Circle */}
                <Box
                  sx={{
                    width: 40,
                    height: 40,
                    borderRadius: '50%',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    border: index < currentStep ? 'none' : '2px solid #CBD5E1',
                    bgcolor: index < currentStep ? '#10B981' : index === currentStep ? '#3B82F6' : '#F1F5F9',
                    color: 'white',
                    fontWeight: 700,
                    fontSize: '14px',
                    transition: 'all 0.4s ease',
                    boxShadow: index === currentStep ? '0 4px 12px rgba(59, 130, 246, 0.4)' : index < currentStep ? '0 4px 12px rgba(16, 185, 129, 0.3)' : 'none',
                    position: 'relative',
                    zIndex: 2,
                    flexShrink: 0,
                    animation: index === currentStep ? 'pulse 1.5s ease-in-out infinite' : 'none',
                  }}
                >
                  {index < currentStep ? (
                    <Check className="w-5 h-5" />
                  ) : index === currentStep ? (
                    <Box
                      sx={{
                        width: 12,
                        height: 12,
                        bgcolor: 'white',
                        borderRadius: '50%',
                        animation: 'ping 1s cubic-bezier(0, 0, 0.2, 1) infinite',
                      }}
                    />
                  ) : (
                    <Box sx={{ color: '#94A3B8', fontWeight: 600 }}>{index + 1}</Box>
                  )}
                </Box>

                {/* Step Label */}
                <Box sx={{ ml: 3, flex: 1, textAlign: 'left' }}>
                  <Box
                    sx={{
                      fontSize: '15px',
                      fontWeight: index <= currentStep ? 700 : 500,
                      color: index <= currentStep ? '#0F172A' : '#94A3B8',
                      transition: 'all 0.3s ease',
                      mb: 0.5,
                    }}
                  >
                    {step.label}
                  </Box>
                  {index === currentStep && (
                    <Box
                      sx={{
                        fontSize: '12px',
                        color: '#3B82F6',
                        fontWeight: 600,
                        animation: 'fadeIn 0.3s ease-in',
                      }}
                    >
                      In Progress...
                    </Box>
                  )}
                  {index < currentStep && (
                    <Box
                      sx={{
                        fontSize: '12px',
                        color: '#10B981',
                        fontWeight: 600,
                      }}
                    >
                      Completed ✓
                    </Box>
                  )}
                </Box>

                {/* Connector Line */}
                {index < loadingSteps.length - 1 && (
                  <Box
                    sx={{
                      position: 'absolute',
                      left: 19,
                      top: 40,
                      width: 2,
                      height: 32,
                      bgcolor: index < currentStep ? '#10B981' : '#E2E8F0',
                      transition: 'all 0.4s ease',
                      zIndex: 1,
                    }}
                  />
                )}
              </Box>
            ))}
          </Box>

          {/* Progress Bar */}
          <Box sx={{ mt: 4, px: 1 }}>
            <Box
              sx={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                mb: 1.5,
              }}
            >
              <Box sx={{ fontSize: '13px', fontWeight: 600, color: '#64748B' }}>
                Loading Progress
              </Box>
              <Box sx={{ fontSize: '14px', fontWeight: 700, color: '#3B82F6' }}>
                {Math.round(progress)}%
              </Box>
            </Box>
            <Box
              sx={{
                height: 12,
                bgcolor: '#F1F5F9',
                borderRadius: '20px',
                overflow: 'hidden',
                border: '1px solid #E2E8F0',
                position: 'relative',
              }}
            >
              <Box
                sx={{
                  height: '100%',
                  width: `${progress}%`,
                  background: 'linear-gradient(90deg, #10B981 0%, #3B82F6 50%, #8B5CF6 100%)',
                  borderRadius: '20px',
                  transition: 'width 0.3s ease',
                  position: 'relative',
                  overflow: 'hidden',
                  '&::after': {
                    content: '""',
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    right: 0,
                    bottom: 0,
                    background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent)',
                    animation: 'shimmerSlide 1.5s infinite',
                  },
                }}
              />
            </Box>
          </Box>
        </Box>

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
          @keyframes shimmerSlide {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(200%); }
          }
          @keyframes fadeIn {
            0% { opacity: 0; }
            100% { opacity: 1; }
          }
          @keyframes ping {
            75%, 100% {
              transform: scale(2);
              opacity: 0;
            }
          }
        `}
      </style>
    </Box>
  );
}

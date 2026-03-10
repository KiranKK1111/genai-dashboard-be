import { X, Info, Sparkles } from 'lucide-react';
import { RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, ResponsiveContainer } from 'recharts';
import { useState } from 'react';
import { Tooltip, Box, Card, Typography, IconButton } from '@mui/material';

interface ChartModalProps {
  isOpen: boolean;
  onClose: () => void;
}

const data = [
  {
    subject: 'Ocean acidification',
    value: 85,
    fullMark: 100,
    description: 'The alteration of land use and land cover, which can affect biodiversity and ecosystem services. The conversion of natural habitats impacts climate regulation.',
    color: '#06B6D4'
  },
  {
    subject: 'Atmosphere',
    value: 72,
    fullMark: 100,
    description: 'Atmospheric aerosol loading affects climate patterns and air quality. Particulate matter in the atmosphere influences solar radiation and precipitation.',
    color: '#8B5CF6'
  },
  {
    subject: 'Biogeochemical flows',
    value: 65,
    fullMark: 100,
    description: 'The cycling of essential nutrients like nitrogen and phosphorus through ecosystems, affecting water quality and terrestrial productivity.',
    color: '#EC4899'
  },
  {
    subject: 'Climate change',
    value: 78,
    fullMark: 100,
    description: 'Long-term shifts in global temperatures and weather patterns, primarily driven by human activities and greenhouse gas emissions.',
    color: '#F59E0B'
  },
  {
    subject: 'Freshwater use',
    value: 68,
    fullMark: 100,
    description: 'The consumption and management of freshwater resources for human needs, agriculture, and industry, affecting water availability and ecosystems.',
    color: '#3B82F6'
  },
  {
    subject: 'Land-system change',
    value: 80,
    fullMark: 100,
    description: 'Modifications to terrestrial ecosystems through deforestation, urbanization, and agricultural expansion, impacting biodiversity and carbon storage.',
    color: '#10B981'
  },
];

const CustomAxisTick = ({ payload, x, y, index }: any) => {
  const dataItem = data.find(d => d.subject === payload.value);
  
  // Define specific positioning for each label based on their position around the hexagon
  const positions = [
    { offsetX: 0, offsetY: -80, anchorX: 85, anchorY: 20 }, // Top (Ocean acidification)
    { offsetX: 95, offsetY: -25, anchorX: 0, anchorY: 20 }, // Top-right (Atmosphere)
    { offsetX: 95, offsetY: 40, anchorX: 0, anchorY: 20 }, // Bottom-right (Biogeochemical)
    { offsetX: 0, offsetY: 85, anchorX: 85, anchorY: 0 }, // Bottom (Climate change)
    { offsetX: -95, offsetY: 40, anchorX: 170, anchorY: 20 }, // Bottom-left (Freshwater)
    { offsetX: -95, offsetY: -25, anchorX: 170, anchorY: 20 }, // Top-left (Land-system)
  ];
  
  const pos = positions[index % positions.length];
  const labelX = x + pos.offsetX;
  const labelY = y + pos.offsetY;

  return (
    <foreignObject
      x={labelX - pos.anchorX}
      y={labelY - pos.anchorY}
      width="170"
      height="50"
    >
      <Tooltip
        title={
          <Box sx={{ p: 1 }}>
            <Typography variant="body2" sx={{ lineHeight: 1.6 }}>
              {dataItem?.description || ''}
            </Typography>
          </Box>
        }
        arrow
        placement="top"
        componentsProps={{
          tooltip: {
            sx: {
              bgcolor: '#1E293B',
              color: 'white',
              boxShadow: '0 10px 40px rgba(0,0,0,0.3)',
              borderRadius: '12px',
              maxWidth: '320px',
              fontSize: '13px',
              padding: '12px 16px',
              '& .MuiTooltip-arrow': {
                color: '#1E293B',
              },
            },
          },
        }}
      >
        <Card
          sx={{
            px: 2.5,
            py: 1.5,
            textAlign: 'center',
            cursor: 'pointer',
            border: '1px solid #E2E8F0',
            bgcolor: 'white',
            boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
            transition: 'all 0.2s ease',
            borderRadius: '8px',
            '&:hover': {
              boxShadow: '0 8px 24px rgba(59, 130, 246, 0.25)',
              borderColor: '#3B82F6',
              transform: 'translateY(-2px)',
            },
          }}
        >
          <Typography
            variant="body2"
            sx={{
              fontWeight: 600,
              color: '#0F172A',
              fontSize: '13px',
              display: 'block',
              whiteSpace: 'nowrap',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
            }}
          >
            {payload.value}
          </Typography>
        </Card>
      </Tooltip>
    </foreignObject>
  );
};

export function ChartModal({ isOpen, onClose }: ChartModalProps) {
  if (!isOpen) return null;

  const averageScore = Math.round(data.reduce((acc, item) => acc + item.value, 0) / data.length);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      {/* Backdrop with blur */}
      <div 
        className="absolute inset-0 bg-gradient-to-br from-slate-900/60 via-blue-900/40 to-emerald-900/50 backdrop-blur-md animate-in fade-in-0 duration-300" 
        onClick={onClose}
      />
      
      {/* Modal */}
      <Box
        sx={{
          position: 'relative',
          bgcolor: 'white',
          borderRadius: '24px',
          boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
          width: '100%',
          maxWidth: '1200px',
          overflow: 'hidden',
          animation: 'zoom-in 0.5s ease-out',
        }}
      >
        {/* Gradient accent bar */}
        <Box sx={{ height: '6px', background: 'linear-gradient(to right, #06B6D4, #3B82F6, #8B5CF6, #EC4899, #F59E0B, #10B981)' }} />
        
        <Box sx={{ p: { xs: 4, md: 6 } }}>
          {/* Header */}
          <Box sx={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', mb: 4 }}>
            <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 2 }}>
              <Box
                sx={{
                  width: 56,
                  height: 56,
                  background: 'linear-gradient(135deg, #3B82F6 0%, #2563EB 100%)',
                  borderRadius: '16px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  boxShadow: '0 10px 30px rgba(59, 130, 246, 0.3)',
                }}
              >
                <Sparkles className="w-7 h-7 text-white" />
              </Box>
              <Box>
                <Typography variant="h4" sx={{ fontWeight: 700, color: '#0F172A', mb: 0.5 }}>
                  Environmental Impact
                </Typography>
                <Typography variant="body2" sx={{ color: '#64748B', mb: 2 }}>
                  Overall Results & Metrics Analysis
                </Typography>
                <Box
                  sx={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: 1,
                    background: 'linear-gradient(to right, #D1FAE5, #DBEAFE)',
                    px: 2,
                    py: 1,
                    borderRadius: '20px',
                    border: '1px solid #A7F3D0',
                  }}
                >
                  <Box
                    sx={{
                      width: 8,
                      height: 8,
                      bgcolor: '#10B981',
                      borderRadius: '50%',
                      animation: 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
                    }}
                  />
                  <Typography variant="caption" sx={{ fontWeight: 600, color: '#0F172A' }}>
                    Average Score: {averageScore}/100
                  </Typography>
                </Box>
              </Box>
            </Box>
            <IconButton
              onClick={onClose}
              sx={{
                '&:hover': {
                  bgcolor: '#F1F5F9',
                  transform: 'rotate(90deg)',
                },
                transition: 'all 0.2s',
              }}
            >
              <X className="w-6 h-6 text-gray-400" />
            </IconButton>
          </Box>

          {/* Info Banner */}
          <Card
            sx={{
              mb: 4,
              background: 'linear-gradient(to right, #EFF6FF, #F5F3FF, #FCE7F3)',
              border: '1px solid #BFDBFE',
              borderRadius: '16px',
              p: 2.5,
              display: 'flex',
              alignItems: 'flex-start',
              gap: 2,
              boxShadow: '0 1px 3px rgba(0,0,0,0.05)',
            }}
          >
            <Box
              sx={{
                width: 40,
                height: 40,
                bgcolor: '#3B82F6',
                borderRadius: '12px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                flexShrink: 0,
                boxShadow: '0 4px 12px rgba(59, 130, 246, 0.3)',
              }}
            >
              <Info className="w-5 h-5 text-white" />
            </Box>
            <Box>
              <Typography variant="subtitle2" sx={{ fontWeight: 600, color: '#0F172A', mb: 0.5 }}>
                Interactive Chart Guide
              </Typography>
              <Typography variant="body2" sx={{ color: '#475569', lineHeight: 1.6 }}>
                Hover over the label boxes around the chart to view detailed descriptions and insights about each environmental metric.
              </Typography>
            </Box>
          </Card>

          {/* Chart Container */}
          <Box
            sx={{
              width: '100%',
              height: '650px',
              background: 'linear-gradient(135deg, #F8FAFC 0%, #EFF6FF 50%, #ECFDF5 100%)',
              borderRadius: '16px',
              p: 6,
              border: '1px solid #E2E8F0',
              boxShadow: 'inset 0 2px 4px rgba(0,0,0,0.06)',
              position: 'relative',
              overflow: 'visible',
            }}
          >
            {/* Decorative background elements */}
            <Box
              sx={{
                position: 'absolute',
                top: 0,
                right: 0,
                width: '256px',
                height: '256px',
                bgcolor: 'rgba(59, 130, 246, 0.03)',
                borderRadius: '50%',
                filter: 'blur(60px)',
              }}
            />
            <Box
              sx={{
                position: 'absolute',
                bottom: 0,
                left: 0,
                width: '256px',
                height: '256px',
                bgcolor: 'rgba(16, 185, 129, 0.03)',
                borderRadius: '50%',
                filter: 'blur(60px)',
              }}
            />
            
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart data={data} margin={{ top: 120, right: 180, bottom: 120, left: 180 }}>
                <defs>
                  <linearGradient id="radarGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#10B981" stopOpacity={0.85} />
                    <stop offset="50%" stopColor="#34D399" stopOpacity={0.7} />
                    <stop offset="100%" stopColor="#6EE7B7" stopOpacity={0.5} />
                  </linearGradient>
                  <filter id="glow">
                    <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
                    <feMerge>
                      <feMergeNode in="coloredBlur"/>
                      <feMergeNode in="SourceGraphic"/>
                    </feMerge>
                  </filter>
                </defs>
                <PolarGrid 
                  strokeDasharray="0" 
                  stroke="#CBD5E1" 
                  strokeWidth={1.5}
                  strokeOpacity={0.4}
                />
                <PolarAngleAxis 
                  dataKey="subject" 
                  tick={<CustomAxisTick />}
                />
                <PolarRadiusAxis 
                  angle={90} 
                  domain={[0, 100]} 
                  tick={{ fill: '#64748B', fontSize: 13, fontWeight: 600 }}
                  axisLine={false}
                  tickCount={6}
                />
                <Radar
                  name="Environmental Metrics"
                  dataKey="value"
                  stroke="#059669"
                  strokeWidth={3}
                  fill="url(#radarGradient)"
                  fillOpacity={0.75}
                  dot={{ 
                    r: 7, 
                    fill: '#10B981', 
                    strokeWidth: 3, 
                    stroke: '#fff',
                    filter: 'url(#glow)'
                  }}
                  activeDot={{
                    r: 9,
                    fill: '#059669',
                    strokeWidth: 4,
                    stroke: '#fff'
                  }}
                />
              </RadarChart>
            </ResponsiveContainer>
          </Box>

          {/* Legend */}
          <Box sx={{ mt: 4, display: 'grid', gridTemplateColumns: { xs: '1fr 1fr', md: '1fr 1fr 1fr' }, gap: 1.5 }}>
            {data.map((item, idx) => (
              <Card
                key={idx}
                sx={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 1,
                  bgcolor: '#F8FAFC',
                  px: 2,
                  py: 1.5,
                  border: '1px solid #E2E8F0',
                  boxShadow: 'none',
                  transition: 'all 0.2s',
                  '&:hover': {
                    borderColor: '#CBD5E1',
                    boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
                  },
                }}
              >
                <Box sx={{ width: 12, height: 12, borderRadius: '50%', bgcolor: item.color, flexShrink: 0 }} />
                <Typography variant="caption" sx={{ fontWeight: 500, color: '#475569', fontSize: '11px', flex: 1 }}>
                  {item.subject}
                </Typography>
                <Typography variant="caption" sx={{ fontWeight: 700, color: '#0F172A', fontSize: '12px' }}>
                  {item.value}
                </Typography>
              </Card>
            ))}
          </Box>
        </Box>
      </Box>
    </div>
  );
}
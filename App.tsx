import { useState } from 'react';
import { ChartModal } from './components/ChartModal';
import { BarChart3, TrendingUp, Activity } from 'lucide-react';

export default function App() {
  const [isModalOpen, setIsModalOpen] = useState(false);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-emerald-50 flex items-center justify-center p-4 relative overflow-hidden">
      {/* Decorative background elements */}
      <div className="absolute top-20 left-10 w-72 h-72 bg-blue-400/10 rounded-full blur-3xl animate-pulse" />
      <div className="absolute bottom-20 right-10 w-96 h-96 bg-emerald-400/10 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '1s' }} />
      
      <div className="text-center relative z-10 max-w-3xl">
        {/* Badge */}
        <div className="inline-flex items-center gap-2 bg-white/80 backdrop-blur-sm px-4 py-2 rounded-full border border-blue-200 mb-6 shadow-sm">
          <Activity className="w-4 h-4 text-blue-600" />
          <span className="text-sm font-semibold text-gray-700">Environmental Analytics Dashboard</span>
        </div>
        
        {/* Main heading */}
        <h1 className="text-5xl md:text-6xl font-bold text-gray-900 mb-4 leading-tight">
          Environmental Impact
          <span className="block bg-gradient-to-r from-blue-600 via-purple-600 to-emerald-600 bg-clip-text text-transparent">
            Assessment Tool
          </span>
        </h1>
        
        <p className="text-lg text-gray-600 mb-10 max-w-2xl mx-auto leading-relaxed">
          Explore comprehensive environmental metrics through interactive radar charts with detailed insights for each impact area
        </p>
        
        {/* CTA Button */}
        <button
          onClick={() => setIsModalOpen(true)}
          className="group relative inline-flex items-center gap-3 px-8 py-4 bg-gradient-to-r from-blue-600 to-blue-700 text-white font-semibold rounded-2xl shadow-xl shadow-blue-500/30 hover:shadow-2xl hover:shadow-blue-500/40 transition-all duration-300 transform hover:scale-105 hover:-translate-y-1"
        >
          <BarChart3 className="w-6 h-6 transition-transform group-hover:rotate-12" />
          <span>View Environmental Radar Chart</span>
          <TrendingUp className="w-5 h-5 transition-transform group-hover:translate-x-1" />
          
          {/* Button shine effect */}
          <div className="absolute inset-0 rounded-2xl bg-gradient-to-r from-transparent via-white/20 to-transparent translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-1000" />
        </button>
        
        {/* Feature highlights */}
        <div className="mt-12 grid grid-cols-1 md:grid-cols-3 gap-4 max-w-2xl mx-auto">
          {[
            { icon: '📊', title: 'Interactive Charts', desc: 'Hover for details' },
            { icon: '🌍', title: '6 Key Metrics', desc: 'Comprehensive analysis' },
            { icon: '💡', title: 'Instant Insights', desc: 'Real-time tooltips' }
          ].map((feature, idx) => (
            <div 
              key={idx} 
              className="bg-white/60 backdrop-blur-sm rounded-xl p-4 border border-gray-200 hover:border-blue-300 transition-all duration-200 hover:shadow-md"
            >
              <div className="text-3xl mb-2">{feature.icon}</div>
              <div className="font-semibold text-gray-900 text-sm">{feature.title}</div>
              <div className="text-xs text-gray-600 mt-1">{feature.desc}</div>
            </div>
          ))}
        </div>
      </div>

      <ChartModal 
        isOpen={isModalOpen} 
        onClose={() => setIsModalOpen(false)} 
      />
    </div>
  );
}
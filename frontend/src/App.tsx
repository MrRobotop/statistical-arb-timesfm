import { useState, useEffect } from 'react';
import { apiClient } from './api/client';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, AreaChart, Area } from 'recharts';
import { Terminal, Activity, ShieldAlert, Cpu, Newspaper, List, BarChart3, FlaskConical, PlayCircle, History, TrendingUp } from 'lucide-react';

const TICKER_DATA = [
  { symbol: 'KO', price: '58.42', change: '+0.45%' },
  { symbol: 'PEP', price: '165.12', change: '-0.12%' },
  { symbol: 'XOM', price: '118.22', change: '+1.22%' },
  { symbol: 'CVX', price: '155.40', change: '+0.85%' },
  { symbol: 'MSFT', price: '425.10', change: '-0.44%' },
  { symbol: 'GOOGL', price: '172.50', change: '+0.15%' },
];

function App() {
  const [data, setData] = useState<any>(null);
  const [backtestData, setBacktestData] = useState<any>(null);
  const [news, setNews] = useState<any[]>([]);
  const [universe, setUniverse] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [initialised, setInitialised] = useState(false);
  const [command, setCommand] = useState('V MA 1.0');
  const [tickerOffset, setTickerOffset] = useState(0);
  const [useKalman, setUseKalman] = useState(true);
  const [activeTab, setActiveTab] = useState<'FORECAST' | 'BACKTEST'>('FORECAST');

  useEffect(() => {
    const timer = setInterval(() => {
      setTickerOffset((prev) => (prev + 1) % 1000);
    }, 50);
    
    const init = async () => {
      try {
        const universeRes = await apiClient.get('/pairs/universe');
        setUniverse(universeRes.data.pairs);
      } catch (e) {
        console.error("Universe fetch failed", e);
      }
      
      try {
        const newsRes = await apiClient.get('/news');
        setNews(newsRes.data.news);
      } catch (e) {
        console.error("Initial news fetch failed", e);
      }
      
      setInitialised(true);
    };
    init();

    return () => clearInterval(timer);
  }, []);

  const runAnalysis = async (cmd?: string) => {
    const currentCmd = cmd || command;
    const parts = currentCmd.split(' ');
    const t1 = parts[0];
    const t2 = parts[1];
    const beta = parts[2] || "1.0";

    setLoading(true);
    try {
      if (activeTab === 'FORECAST') {
        const res = await apiClient.post('/forecast/spread', {
          ticker_a: t1,
          ticker_b: t2,
          hedge_ratio: parseFloat(beta),
          context_days: 252,
          horizon_days: 30,
          use_kalman: useKalman
        });
        setData(res.data);
      } else {
        const res = await apiClient.post('/backtest/run', {
          ticker_a: t1,
          ticker_b: t2,
          hedge_ratio: parseFloat(beta),
          start_date: "2024-04-20",
          end_date: new Date().toISOString().split('T')[0],
          use_kalman: useKalman,
          notional: 10000
        });
        setBacktestData(res.data);
      }
      
      const newsRes = await apiClient.get('/news', {
        params: { tickers: [t1, t2] }
      });
      setNews(newsRes.data.news);
    } catch (e: any) {
      console.error(e);
      alert(e.response?.data?.detail || "Action failed. Ensure tickers are valid.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-bloomberg-bg text-bloomberg-text font-mono flex flex-col overflow-hidden">
      {/* Bloomberg Ticker Tape */}
      <div className="bg-bloomberg-gray/20 border-b border-bloomberg-text h-8 flex items-center overflow-hidden whitespace-nowrap">
        <div 
          className="flex gap-8 px-4 transition-transform duration-1000 ease-linear"
          style={{ transform: `translateX(-${tickerOffset}px)` }}
        >
          {[...TICKER_DATA, ...TICKER_DATA, ...TICKER_DATA, ...TICKER_DATA, ...TICKER_DATA].map((t, i) => (
            <div key={i} className="flex gap-2 text-xs">
              <span className="text-bloomberg-blue font-bold">{t.symbol}</span>
              <span className="text-[#CCC]">{t.price}</span>
              <span className={t.change.startsWith('+') ? 'text-bloomberg-green' : 'text-bloomberg-red'}>
                {t.change}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Main Terminal Header */}
      <header className="bg-bloomberg-bg border-b border-bloomberg-text p-2 flex justify-between items-center px-4">
        <div className="flex items-center gap-4">
          <Terminal size={24} className="text-bloomberg-text" />
          <h1 className="text-lg font-bold tracking-tighter flex items-center gap-2">
            PAIRS TRADER <span className="bg-bloomberg-text text-black px-1">V2.6</span>
            <span className="text-bloomberg-blue ml-2">STAT-ARB TERMINAL</span>
          </h1>
        </div>
        <div className="flex items-center gap-6 text-xs">
          <button 
            onClick={() => setUseKalman(!useKalman)}
            className={`flex items-center gap-2 px-2 py-1 border ${useKalman ? 'border-bloomberg-green text-bloomberg-green bg-bloomberg-green/10' : 'border-bloomberg-gray text-bloomberg-gray'}`}
          >
            <FlaskConical size={14} />
            <span>KALMAN FILTER: {useKalman ? 'ON' : 'OFF'}</span>
          </button>
          <div className="flex items-center gap-2">
            <Cpu size={14} className="text-bloomberg-blue" />
            <span>CORE: <span className="text-bloomberg-blue font-bold">TIMESFM_2.5</span></span>
          </div>
          <div className="flex items-center gap-2 text-bloomberg-green">
            <Activity size={14} />
            <span>SERVER: {initialised ? 'ONLINE' : 'BOOTING'}</span>
          </div>
        </div>
      </header>

      {/* Command Input Bar */}
      <div className="bg-bloomberg-gray/20 p-1 flex items-center gap-2 border-b border-bloomberg-text px-4">
        <div className="flex gap-1 mr-4">
          <button 
            onClick={() => setActiveTab('FORECAST')}
            className={`px-3 py-1 text-[10px] font-bold uppercase transition-colors ${activeTab === 'FORECAST' ? 'bg-bloomberg-text text-black' : 'text-bloomberg-gray hover:text-bloomberg-text'}`}
          >
            Forecast
          </button>
          <button 
            onClick={() => setActiveTab('BACKTEST')}
            className={`px-3 py-1 text-[10px] font-bold uppercase transition-colors ${activeTab === 'BACKTEST' ? 'bg-bloomberg-text text-black' : 'text-bloomberg-gray hover:text-bloomberg-text'}`}
          >
            Backtest
          </button>
        </div>
        <span className="text-bloomberg-green font-bold text-sm">{'>>'}</span>
        <input 
          type="text" 
          value={command}
          onChange={(e) => setCommand(e.target.value.toUpperCase())}
          onKeyDown={(e) => e.key === 'Enter' && runAnalysis()}
          placeholder="ENTER TICKERS AND BETA (e.g. MSFT GOOGL 1.1)"
          className="bg-transparent border-none outline-none text-bloomberg-text w-full text-sm placeholder:text-bloomberg-gray/50"
        />
        <button 
          onClick={() => runAnalysis()}
          disabled={loading}
          className="hover:bg-bloomberg-text hover:text-black px-4 py-1 transition-colors uppercase text-xs font-bold border border-bloomberg-text"
        >
          {loading ? 'BUSY' : activeTab === 'FORECAST' ? 'ANALYSE' : 'RUN BT'}
        </button>
      </div>

      <main className="flex-1 p-4 grid grid-cols-12 gap-4 overflow-hidden">
        {/* Left: Monitor & Signal Panel */}
        <div className="col-span-12 lg:col-span-3 flex flex-col gap-4 overflow-hidden">
          
          {/* PAIR UNIVERSE / MONITOR */}
          <section className="border border-bloomberg-text flex flex-col min-h-0">
            <h2 className="text-[10px] bg-bloomberg-blue text-black px-1 font-bold uppercase flex items-center gap-1">
              <List size={10} /> Universe Monitor
            </h2>
            <div className="flex-1 overflow-auto p-2">
              <table className="w-full text-[10px] text-left">
                <thead>
                  <tr className="text-bloomberg-gray border-b border-bloomberg-gray uppercase font-bold">
                    <th className="pb-1">Pair</th>
                    <th className="pb-1">Sector</th>
                    <th className="pb-1 text-right">Action</th>
                  </tr>
                </thead>
                <tbody>
                  {universe.map((pair, idx) => (
                    <tr key={idx} className="border-b border-bloomberg-gray/20 hover:bg-bloomberg-gray/10 cursor-pointer" 
                        onClick={() => { setCommand(`${pair.ticker_a} ${pair.ticker_b} 1.0`); runAnalysis(`${pair.ticker_a} ${pair.ticker_b} 1.0`); }}>
                      <td className="py-1.5 font-bold text-white">{pair.ticker_a}/{pair.ticker_b}</td>
                      <td className="py-1.5 text-bloomberg-gray">{pair.sector}</td>
                      <td className="py-1.5 text-right text-bloomberg-blue underline">LOAD</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          {/* DYNAMIC METRICS / SIGNAL */}
          <section className="border border-bloomberg-text p-3">
            <h2 className="text-[10px] bg-bloomberg-text text-black px-1 font-bold mb-3 uppercase flex items-center gap-1">
              {activeTab === 'FORECAST' ? <ShieldAlert size={10} /> : <TrendingUp size={10} />}
              {activeTab === 'FORECAST' ? 'Strategy Signal' : 'BT Performance'}
            </h2>
            
            {activeTab === 'FORECAST' ? (
              data ? (
                <div className="flex flex-col gap-4">
                  <div className="flex justify-between items-end border-b border-bloomberg-gray/30 pb-2">
                    <span className="text-[10px] uppercase text-bloomberg-gray font-bold">Signal</span>
                    <div className={`text-2xl font-bold ${data.signal.action.includes('BUY') ? 'text-bloomberg-green' : data.signal.action.includes('SELL') ? 'text-bloomberg-red' : 'text-bloomberg-text'}`}>
                      {data.signal.action}
                    </div>
                  </div>
                  <div className="flex justify-between items-center text-sm">
                    <span className="uppercase text-bloomberg-gray font-bold text-[10px]">Confidence</span>
                    <span className="font-bold text-white">{(data.signal.confidence * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between items-center text-sm">
                    <span className="uppercase text-bloomberg-gray font-bold text-[10px]">Current Z</span>
                    <span className={`font-bold ${Math.abs(data.signal.z_score) > 2 ? 'text-bloomberg-red' : 'text-bloomberg-green'}`}>
                      {data.signal.z_score.toFixed(3)}
                    </span>
                  </div>
                </div>
              ) : (
                <div className="h-24 flex items-center justify-center text-bloomberg-gray text-[10px] italic">
                  {loading ? 'CALCULATING MODELS...' : 'SELECT PAIR FROM MONITOR'}
                </div>
              )
            ) : (
              backtestData ? (
                <div className="flex flex-col gap-3 text-[11px]">
                  <div className="flex justify-between border-b border-bloomberg-gray/20 pb-1">
                    <span className="text-bloomberg-gray uppercase font-bold">Ann. Return</span>
                    <span className="text-bloomberg-green font-bold">{(backtestData.metrics.annualized_return * 100).toFixed(2)}%</span>
                  </div>
                  <div className="flex justify-between border-b border-bloomberg-gray/20 pb-1">
                    <span className="text-bloomberg-gray uppercase font-bold">Sharpe Ratio</span>
                    <span className="text-white font-bold">{backtestData.metrics.sharpe_ratio.toFixed(2)}</span>
                  </div>
                  <div className="flex justify-between border-b border-bloomberg-gray/20 pb-1">
                    <span className="text-bloomberg-gray uppercase font-bold">Win Rate</span>
                    <span className="text-bloomberg-blue font-bold">{(backtestData.metrics.win_rate * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between border-b border-bloomberg-gray/20 pb-1">
                    <span className="text-bloomberg-gray uppercase font-bold">Max Drawdown</span>
                    <span className="text-bloomberg-red font-bold">{(backtestData.metrics.max_drawdown * 100).toFixed(2)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-bloomberg-gray uppercase font-bold">Trades</span>
                    <span className="text-white font-bold">{backtestData.metrics.num_trades}</span>
                  </div>
                </div>
              ) : (
                <div className="h-24 flex items-center justify-center text-bloomberg-gray text-[10px] italic text-center">
                  {loading ? 'SIMULATING TRADES...' : 'RUN BACKTEST TO VIEW METRICS'}
                </div>
              )
            )}
          </section>

          {/* WORLD NEWS */}
          <section className="border border-bloomberg-text flex flex-col min-h-0 flex-1">
            <h2 className="text-[10px] bg-white text-black px-1 font-bold uppercase flex items-center gap-1">
              <Newspaper size={10} /> Live News Feed
            </h2>
            <div className="flex-1 overflow-auto p-2 flex flex-col gap-3">
              {news.length > 0 ? (
                news.map((item, idx) => (
                  <div key={idx} className="border-b border-bloomberg-gray/20 pb-2 last:border-0">
                    <div className="flex justify-between items-start gap-2 mb-1">
                      <span className="text-[10px] text-bloomberg-blue font-bold px-1 border border-bloomberg-blue">{item.ticker}</span>
                      <span className={`text-[10px] font-bold ${item.sentiment === 'POSITIVE' ? 'text-bloomberg-green' : item.sentiment === 'NEGATIVE' ? 'text-bloomberg-red' : 'text-bloomberg-gray'}`}>
                        {item.sentiment}
                      </span>
                    </div>
                    <a href={item.link} target="_blank" rel="noreferrer" className="text-[11px] text-[#EEE] hover:text-bloomberg-blue line-clamp-2 block leading-tight">
                      {item.title}
                    </a>
                  </div>
                ))
              ) : (
                <div className="h-24 flex flex-col items-center justify-center gap-2 opacity-30">
                  <Activity size={24} />
                  <span className="text-[10px]">SYNCING NEWS...</span>
                </div>
              )}
            </div>
          </section>

        </div>

        {/* Center/Right: Visuals */}
        <div className="col-span-12 lg:col-span-9 flex flex-col gap-4 overflow-hidden">
          <div className="border border-bloomberg-text p-4 flex-1 flex flex-col min-h-[500px]">
            {activeTab === 'FORECAST' ? (
              <>
                <div className="flex justify-between items-start mb-4">
                  <h2 className="text-xs bg-bloomberg-text text-black px-1 font-bold inline-block uppercase flex items-center gap-1">
                    <BarChart3 size={12} /> Spread & TimesFM Projection {useKalman && <span className="ml-2 font-normal opacity-70">(KALMAN MODE)</span>}
                  </h2>
                  {data && (
                    <div className="flex gap-4 text-[10px] bg-bloomberg-gray/10 px-2 py-0.5 border border-bloomberg-gray/20">
                      <div className="flex items-center gap-1">
                        <div className="w-3 h-0.5 bg-bloomberg-blue"></div>
                        <span className="text-bloomberg-gray font-bold">HISTORICAL SPREAD</span>
                      </div>
                      <div className="flex items-center gap-1">
                        <div className="w-3 h-0.5 bg-bloomberg-green border-dashed"></div>
                        <span className="text-bloomberg-gray font-bold">TIMESFM PREDICTION</span>
                      </div>
                    </div>
                  )}
                </div>
                
                <div className="flex-1 bg-black/40 border border-bloomberg-gray/10">
                  {data ? (
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={[
                        ...data.spread_history.map((val: any, idx: number) => ({ 
                          day: idx - 252, 
                          spread: val 
                        })),
                        ...data.forecast.point.map((val: any, idx: number) => ({ 
                          day: idx + 1, 
                          forecast: val 
                        }))
                      ]} margin={{ top: 20, right: 30, left: 0, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#333" vertical={false} />
                        <XAxis dataKey="day" stroke="#999" fontSize={10} tick={{ fill: '#999' }} axisLine={{ stroke: '#444' }} />
                        <YAxis stroke="#999" fontSize={10} domain={['auto', 'auto']} tick={{ fill: '#999' }} axisLine={{ stroke: '#444' }} />
                        <Tooltip 
                          contentStyle={{ backgroundColor: '#000', border: '1px solid #FFB800', color: '#FFB800', fontFamily: 'monospace' }}
                          itemStyle={{ fontSize: '11px', padding: '0px' }}
                        />
                        <ReferenceLine x={0} stroke="#FFB800" strokeDasharray="5 5" label={{ value: 'MARKET_NOW', position: 'insideTopRight', fill: '#FFB800', fontSize: 10, fontWeight: 'bold' }} />
                        <ReferenceLine y={0} stroke="#555" strokeWidth={1} />
                        <Line type="monotone" dataKey="spread" stroke="#00FFFF" dot={false} strokeWidth={1.5} name="History" animationDuration={500} />
                        <Line type="monotone" dataKey="forecast" stroke="#00FF00" strokeDasharray="5 5" dot={false} strokeWidth={2} name="TimesFM" />
                      </LineChart>
                    </ResponsiveContainer>
                  ) : (
                    <WelcomeView initialised={initialised} universe={universe} />
                  )}
                </div>
              </>
            ) : (
              <>
                <div className="flex justify-between items-start mb-4">
                  <h2 className="text-xs bg-bloomberg-blue text-black px-1 font-bold inline-block uppercase flex items-center gap-1">
                    <History size={12} /> 2-Year Equity Curve {useKalman && <span className="ml-2 font-normal opacity-70">(KALMAN MODE)</span>}
                  </h2>
                  {backtestData && (
                    <div className="text-[10px] text-bloomberg-gray font-bold uppercase">
                      Total Net PnL: <span className="text-bloomberg-green font-bold ml-1">
                        ${backtestData.trades.reduce((sum: number, t: any) => sum + t.net_pnl, 0).toLocaleString()}
                      </span>
                    </div>
                  )}
                </div>
                
                <div className="flex-1 bg-black/40 border border-bloomberg-gray/10">
                  {backtestData ? (
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={backtestData.equity_curve.map((val: any, idx: number) => ({ 
                        date: backtestData.equity_dates[idx], 
                        equity: val 
                      }))} margin={{ top: 20, right: 30, left: 10, bottom: 0 }}>
                        <defs>
                          <linearGradient id="colorEq" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#00FF00" stopOpacity={0.3}/>
                            <stop offset="95%" stopColor="#00FF00" stopOpacity={0}/>
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="#333" vertical={false} />
                        <XAxis dataKey="date" stroke="#999" fontSize={9} tick={{ fill: '#666' }} axisLine={{ stroke: '#444' }} minTickGap={50} />
                        <YAxis stroke="#999" fontSize={10} domain={['dataMin - 100', 'dataMax + 100']} tick={{ fill: '#999' }} axisLine={{ stroke: '#444' }} />
                        <Tooltip 
                          contentStyle={{ backgroundColor: '#000', border: '1px solid #00FF00', color: '#00FF00', fontFamily: 'monospace' }}
                          labelStyle={{ color: '#888', marginBottom: '4px' }}
                        />
                        <Area type="monotone" dataKey="equity" stroke="#00FF00" fillOpacity={1} fill="url(#colorEq)" strokeWidth={2} name="Account Value" />
                      </AreaChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="h-full flex flex-col items-center justify-center gap-4 text-center px-20">
                      <PlayCircle size={48} className="text-bloomberg-blue animate-pulse" />
                      <div>
                        <div className="text-bloomberg-text text-lg font-bold uppercase mb-2">Backtest Terminal</div>
                        <p className="text-bloomberg-gray text-xs leading-relaxed max-w-md mx-auto">
                          Simulate the statistical arbitrage strategy over the past 2 years. 
                          Includes transaction costs, slippage, and optional Kalman Filter regime adaptation.
                        </p>
                      </div>
                      <button 
                        onClick={() => runAnalysis()}
                        className="mt-4 border border-bloomberg-blue text-bloomberg-blue hover:bg-bloomberg-blue hover:text-black px-8 py-2 font-bold transition-all uppercase text-sm"
                      >
                        Launch Simulation
                      </button>
                    </div>
                  )}
                </div>
              </>
            )}
          </div>
        </div>
      </main>

      {/* Status Bar */}
      <footer className="bg-bloomberg-text text-black text-[10px] px-4 py-1 flex justify-between font-bold border-t border-black">
        <div>TERMINAL ID: TRADER-8492 | AUTH: {initialised ? 'VALID' : 'PENDING'}</div>
        <div className="flex gap-4">
          <span className="flex items-center gap-1">MODE: <span className="underline">{useKalman ? 'KALMAN_DYNAMIC' : 'OLS_STATIC'}</span></span>
          <span>SENTIMENT_PRO: prosus-finbert</span>
          <span>GPU: {initialised ? 'ACTIVE' : 'IDLE'}</span>
          <span>{new Date().toLocaleTimeString()}</span>
        </div>
      </footer>
    </div>
  );
}

function WelcomeView({ initialised, universe }: { initialised: boolean, universe: any[] }) {
  return (
    <div className="h-full flex flex-col items-center justify-center gap-6">
      <div className="relative">
        <ShieldAlert size={64} className="text-bloomberg-gray/30 animate-pulse" />
        {!initialised && <div className="absolute inset-0 flex items-center justify-center text-[10px] font-bold text-white uppercase">Loading</div>}
      </div>
      <div className="text-center flex flex-col gap-2">
        <span className="text-bloomberg-text text-sm tracking-[0.3em] font-bold uppercase">PairsTrader Link v2.6</span>
        <span className="text-bloomberg-gray text-[10px] uppercase font-bold">
          {initialised ? 'Ready for command entry...' : 'Booting TimesFM 2.5 Inference Engine...'}
        </span>
      </div>
      {initialised && (
        <div className="grid grid-cols-2 gap-4 mt-4">
           <div className="border border-bloomberg-gray/30 p-3 text-[10px] max-w-[200px] bg-black/20">
              <span className="text-bloomberg-blue block font-bold mb-1 uppercase">Universe Status</span>
              {universe.length} high-conviction pairs validated. Select a pair to begin analysis.
           </div>
           <div className="border border-bloomberg-gray/30 p-3 text-[10px] max-w-[200px] bg-black/20">
              <span className="text-bloomberg-green block font-bold mb-1 uppercase">Machine Mode</span>
              Kalman Filter active for dynamic regime adaptation. TimesFM forecast enabled.
           </div>
        </div>
      )}
    </div>
  );
}

export default App;

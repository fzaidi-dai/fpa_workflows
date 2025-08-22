#!/usr/bin/env python3
"""
Deployment Configuration for Category-Wise Formula MCP Servers

This script manages the deployment of focused, category-wise formula servers
that provide manageable toolsets for AI agents.
"""

import asyncio
import subprocess
import time
import signal
import sys
import os
from typing import Dict, List
import json

# Server configurations
CATEGORY_SERVERS = {
    'aggregation': {
        'name': 'Aggregation Formula Server',
        'script': 'aggregation_formula_mcp_server.py',
        'port': 3030,
        'description': 'SUM, SUMIF, COUNT, AVERAGE formulas',
        'tools': ['sum', 'average', 'count', 'sumif', 'sumifs', 'countif', 'countifs', 'averageif', 'averageifs', 'subtotal', 'counta', 'max', 'min']
    },
    'lookup': {
        'name': 'Lookup Formula Server',
        'script': 'lookup_formula_mcp_server.py', 
        'port': 3031,
        'description': 'VLOOKUP, INDEX/MATCH, XLOOKUP formulas',
        'tools': ['vlookup', 'hlookup', 'xlookup', 'index', 'match', 'index_match']
    },
    'financial': {
        'name': 'Financial Formula Server',
        'script': 'financial_formula_mcp_server.py',
        'port': 3032,
        'description': 'NPV, IRR, PMT, depreciation formulas',
        'tools': ['npv', 'irr', 'mirr', 'xirr', 'xnpv', 'pmt', 'pv', 'fv', 'nper', 'rate', 'ipmt', 'ppmt', 'sln', 'ddb']
    },
    'array': {
        'name': 'Array Formula Server',
        'script': 'array_formula_mcp_server.py',
        'port': 3034,
        'description': 'ARRAYFORMULA, TRANSPOSE, UNIQUE, SORT formulas',
        'tools': ['arrayformula', 'transpose', 'unique', 'sort', 'filter', 'sequence', 'sumproduct']
    },
    'text': {
        'name': 'Text Formula Server',
        'script': 'text_formula_mcp_server.py',
        'port': 3035,
        'description': 'CONCATENATE, LEFT, RIGHT, text formulas',
        'tools': ['concatenate', 'left', 'right', 'mid', 'len', 'upper', 'lower', 'trim']
    },
    'logical': {
        'name': 'Logical Formula Server',
        'script': 'logical_formula_mcp_server.py',
        'port': 3036,
        'description': 'IF, AND, OR, NOT formulas',
        'tools': ['if', 'and', 'or', 'not']
    },
    'statistical': {
        'name': 'Statistical Formula Server',
        'script': 'statistical_formula_mcp_server.py',
        'port': 3037,
        'description': 'MEDIAN, STDEV, VAR, PERCENTILE formulas',
        'tools': ['median', 'stdev', 'var', 'mode', 'percentile', 'percentrank', 'rank']
    },
    'datetime': {
        'name': 'DateTime Formula Server',
        'script': 'datetime_formula_mcp_server.py',
        'port': 3038,
        'description': 'NOW, TODAY, DATE, YEAR formulas',
        'tools': ['now', 'today', 'date', 'year', 'month', 'day', 'eomonth']
    },
    'business': {
        'name': 'Business Formula Server',
        'script': 'business_formula_mcp_server.py',
        'port': 3033,
        'description': 'CAGR, Customer LTV, CAPM, business metrics',
        'tools': ['profit_margin', 'cagr', 'customer_ltv', 'churn_rate', 'variance_percent', 'compound_growth', 'capm', 'sharpe_ratio', 'beta_coefficient', 'market_share', 'customer_acquisition_cost', 'break_even_analysis', 'dupont_analysis', 'z_score']
    }
}


class CategoryServerManager:
    """Manages deployment and monitoring of category-wise formula servers"""
    
    def __init__(self):
        self.processes: Dict[str, subprocess.Popen] = {}
        self.base_dir = os.path.join(os.path.dirname(__file__), 'mcp_tooling')
        
    def start_server(self, category: str) -> bool:
        """Start a specific category server"""
        if category not in CATEGORY_SERVERS:
            print(f"❌ Unknown server category: {category}")
            return False
        
        config = CATEGORY_SERVERS[category]
        script_path = os.path.join(self.base_dir, config['script'])
        
        if not os.path.exists(script_path):
            print(f"❌ Server script not found: {script_path}")
            return False
        
        try:
            # Start server with FastAPI on specified port
            cmd = ['uv', 'run', 'python', script_path, '--port', str(config['port'])]
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=os.path.dirname(__file__)
            )
            
            self.processes[category] = process
            print(f"🚀 Started {config['name']} on port {config['port']}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start {category} server: {e}")
            return False
    
    def stop_server(self, category: str) -> bool:
        """Stop a specific category server"""
        if category not in self.processes:
            print(f"⚠️  {category} server not running")
            return True
        
        try:
            process = self.processes[category]
            process.terminate()
            process.wait(timeout=5)
            del self.processes[category]
            print(f"🛑 Stopped {CATEGORY_SERVERS[category]['name']}")
            return True
            
        except subprocess.TimeoutExpired:
            process.kill()
            del self.processes[category]
            print(f"🔪 Force-killed {CATEGORY_SERVERS[category]['name']}")
            return True
        except Exception as e:
            print(f"❌ Error stopping {category} server: {e}")
            return False
    
    def start_all_servers(self) -> bool:
        """Start all category servers"""
        print("🚀 Starting all category-wise formula servers...")
        print("=" * 60)
        
        success_count = 0
        for category in CATEGORY_SERVERS:
            if self.start_server(category):
                success_count += 1
                time.sleep(2)  # Stagger startup
        
        print("=" * 60)
        if success_count == len(CATEGORY_SERVERS):
            print(f"✅ All {len(CATEGORY_SERVERS)} servers started successfully!")
            self.print_server_status()
            return True
        else:
            print(f"⚠️  Only {success_count}/{len(CATEGORY_SERVERS)} servers started")
            return False
    
    def stop_all_servers(self) -> bool:
        """Stop all category servers"""
        print("🛑 Stopping all category servers...")
        
        success_count = 0
        for category in list(self.processes.keys()):
            if self.stop_server(category):
                success_count += 1
        
        if success_count > 0:
            print(f"✅ Stopped {success_count} servers")
        return True
    
    def check_server_health(self, category: str) -> bool:
        """Check if a server is healthy"""
        if category not in self.processes:
            return False
        
        process = self.processes[category]
        return process.poll() is None
    
    def print_server_status(self):
        """Print status of all servers"""
        print("\n📊 Server Status:")
        print("-" * 80)
        print(f"{'Category':<12} {'Name':<25} {'Port':<6} {'Status':<8} {'Tools':<5}")
        print("-" * 80)
        
        for category, config in CATEGORY_SERVERS.items():
            status = "🟢 UP" if self.check_server_health(category) else "🔴 DOWN"
            tool_count = len(config['tools'])
            print(f"{category:<12} {config['name']:<25} {config['port']:<6} {status:<8} {tool_count:<5}")
        
        print("-" * 80)
        print(f"Total Tools Distributed: {sum(len(config['tools']) for config in CATEGORY_SERVERS.values())}")
        print()
    
    def generate_mcp_config(self) -> Dict:
        """Generate MCP configuration for all servers"""
        mcp_config = {
            "mcpServers": {},
            "description": "Category-wise Formula Servers with 100% accuracy guarantee",
            "version": "1.0.0"
        }
        
        for category, config in CATEGORY_SERVERS.items():
            mcp_config["mcpServers"][f"formula_{category}"] = {
                "command": "uv",
                "args": [
                    "run", "python", 
                    f"mcp_tooling/{config['script']}"
                ],
                "env": {},
                "description": f"{config['description']} - {len(config['tools'])} tools",
                "categories": [category],
                "tools": config['tools']
            }
        
        return mcp_config
    
    def save_mcp_config(self, filename: str = "formula_servers_mcp_config.json"):
        """Save MCP configuration to file"""
        config = self.generate_mcp_config()
        
        config_path = os.path.join(os.path.dirname(__file__), filename)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"💾 MCP configuration saved to: {filename}")
        return config_path


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n\n🛑 Shutting down all servers...")
    if 'manager' in globals():
        manager.stop_all_servers()
    sys.exit(0)


async def run_deployment_test():
    """Run a quick deployment test"""
    print("🧪 Running deployment test...")
    
    manager = CategoryServerManager()
    
    # Start all servers
    if not manager.start_all_servers():
        print("❌ Deployment test failed - servers didn't start")
        return False
    
    # Wait for servers to initialize
    print("⏳ Waiting for servers to initialize...")
    await asyncio.sleep(5)
    
    # Check health
    all_healthy = True
    for category in CATEGORY_SERVERS:
        if not manager.check_server_health(category):
            print(f"❌ {category} server is not healthy")
            all_healthy = False
    
    if all_healthy:
        print("✅ All servers are healthy")
        
        # Generate MCP config
        config_path = manager.save_mcp_config()
        print(f"✅ MCP configuration generated: {config_path}")
        
        # Stop servers
        manager.stop_all_servers()
        print("✅ Deployment test completed successfully")
        return True
    else:
        manager.stop_all_servers()
        print("❌ Deployment test failed - unhealthy servers")
        return False


def main():
    """Main deployment management interface"""
    global manager
    
    manager = CategoryServerManager()
    signal.signal(signal.SIGINT, signal_handler)
    
    if len(sys.argv) < 2:
        print("📋 Category-Wise Formula Server Deployment Manager")
        print("=" * 60)
        print("Usage:")
        print("  python deploy_category_servers.py start          - Start all servers")
        print("  python deploy_category_servers.py stop           - Stop all servers")
        print("  python deploy_category_servers.py status         - Show server status")
        print("  python deploy_category_servers.py config         - Generate MCP config")
        print("  python deploy_category_servers.py test           - Run deployment test")
        print("  python deploy_category_servers.py start <category> - Start specific server")
        print("")
        print("Available categories: " + ", ".join(CATEGORY_SERVERS.keys()))
        return
    
    command = sys.argv[1].lower()
    
    if command == "start":
        if len(sys.argv) > 2:
            # Start specific server
            category = sys.argv[2]
            manager.start_server(category)
        else:
            # Start all servers
            manager.start_all_servers()
            
            # Keep running and monitoring
            try:
                print("\n💡 Press Ctrl+C to stop all servers")
                while True:
                    time.sleep(10)
                    # Check if any server died
                    dead_servers = []
                    for category in CATEGORY_SERVERS:
                        if category in manager.processes and not manager.check_server_health(category):
                            dead_servers.append(category)
                    
                    if dead_servers:
                        print(f"⚠️  Dead servers detected: {dead_servers}")
                        for category in dead_servers:
                            print(f"🔄 Restarting {category} server...")
                            manager.stop_server(category)
                            manager.start_server(category)
            except KeyboardInterrupt:
                signal_handler(None, None)
    
    elif command == "stop":
        manager.stop_all_servers()
    
    elif command == "status":
        manager.print_server_status()
    
    elif command == "config":
        config_path = manager.save_mcp_config()
        print(f"MCP configuration saved to: {config_path}")
    
    elif command == "test":
        success = asyncio.run(run_deployment_test())
        sys.exit(0 if success else 1)
    
    else:
        print(f"❌ Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
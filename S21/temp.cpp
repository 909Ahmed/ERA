#include <bits/stdc++.h>
using namespace std;

int main(){
    vector<int> dist(N1 + 1, INT_MIN);
    dist[1] = 0;
    queue<int> q;
    q.push(1);
    while (!q.empty()) {
        
        int u = q.front();
        q.pop();
        
        for (int v : g[u]) {
            
            if (dist[v] == INT_MIN || dist[v] < dist[u] + 1) {
                
                dist[v] = dist[u] + 1;
                q.push(v);
                if(v == N1) pos1.pb(dist[v]);
                
            }
            
        }
        
    }
}
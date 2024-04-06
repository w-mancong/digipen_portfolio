const fs = require('node:fs/promises');

function appendText(word, shouldCensor)
{
    if (shouldCensor)
    {
        return '*'.repeat(word.length) + " ";
    }
    return word + " ";
}

async function filterText(msg)
{
    try {
        const reEscape = s => s.replace(/[-\/\\^$*+?.()|[\]{}]/g, '\\$&');       
        const data = await fs.readFile('profanity_words.txt', { encoding: 'utf8' });
        const badWords = data.split('\n').map(word => word.trim());
        const badWordsRE = new RegExp(badWords.map(reEscape).join('|'));
        
        var words = msg.toLowerCase().split(/\s+/);
        var filteredText = "";
        for (const w of words)
        {
            filteredText += appendText(w, w.match(badWordsRE));
        }
        return filteredText;
    } catch (err) {
        console.log(err);
        return "";
    }
}

async function filterProfanity(msg)
{
    try {
        const filteredText = await filterText(msg);
        console.log(filteredText);
    } catch (err) {
        console.log(err);
    }
}

filterProfanity("");